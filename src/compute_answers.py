""" compute_answers.py 
a python script that given a json file formatted as the training set, creates a prediction file in the desired format. Obviously the given json file will not contain the answers, only the contexts and the questions.

python3 compute_answers.py *path_to_json_file*

The prediction will be saved inside the folder "predictions"

"""

import argparse
import json

import numpy as np
import pandas as pd

from transformers import AutoTokenizer
from transformers import DefaultDataCollator
from transformers import TFAutoModelForQuestionAnswering
from datasets import Dataset
from tqdm.auto import tqdm

import collections



def flatten_input(input_data):
    """Flattens the input data"""
    
    flattened_data = []

    for article in input_data['data']:
        # article is a dictionary with keys: title, paragraphs
        title = article['title'].strip()

        for paragraph in article['paragraphs']:
            # paragraph is a dictionary with keys: context, qas
            context = paragraph['context'].strip()

            for qa in paragraph["qas"]:
                # qa is a dictionary with keys: answers, question, id
                question = qa["question"].strip()
                id_ = qa["id"]

                data.append({'title': title,
                            'context': context,
                            'question': question,
                            'id': id_})
    return flattened_data

def prepare_features(examples, max_len):

    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_len,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1

        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])]
            
    return tokenized_examples

def postprocess_qa_predictions(examples, 
                               features, 
                               raw_predictions, 
                               n_best_size=20, 
                               max_answer_length=30):
    
    all_start_logits = raw_predictions["start_logits"]
    all_end_logits = raw_predictions["end_logits"]

    # Build a map example to its corresponding features. 
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])} 
    features_per_example = collections.defaultdict(list) 

    for i, feature in enumerate(features): 
        features_per_example[example_id_to_index[feature["example_id"]]].append(i) 
 
    # The dictionaries we have to fill. 
    predictions = collections.OrderedDict() 
    
    # Logging
    print(f"Post-processing {len(examples)} example predictions split into {len(features)} features.") 
 
    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)): 
        # Those are the indices of the features associated to the current example. 
        feature_indices = features_per_example[example_index] 
 
        valid_answers = [] 
 
        context = example["context"] 
        # Looping through all the features associated to the current example. 
        for feature_index in feature_indices: 
            # We grab the predictions of the model for this feature. 
            start_logits = all_start_logits[feature_index] 
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions in our logits to span of texts in the original 
            # context. 
            offset_mapping = features[feature_index]["offset_mapping"] 
 
            # Go through all possibilities for the n_best_size greater start and end logits. 
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist() 
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist() 

            for start_index in start_indexes: 
                for end_index in end_indexes: 
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond 
                    # to part of the input_ids that are not in the context. 
                    if (start_index >= len(offset_mapping) 
                        or end_index >= len(offset_mapping) 
                        or offset_mapping[start_index] is None 
                        or offset_mapping[end_index] is None): 
                        
                        continue 

                    # Don't consider answers with a length that is either < 0 or > max_answer_length. 
                    if (end_index < start_index 
                        or end_index - start_index + 1 > max_answer_length): 
                        
                        continue 
 
                    start_char = offset_mapping[start_index][0] 
                    end_char = offset_mapping[end_index][1] 
                    valid_answers.append( 
                        { 
                            "score": start_logits[start_index] + end_logits[end_index], 
                            "text": context[start_char:end_char], 
                        } 
                    ) 
 
        if len(valid_answers) > 0: 
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0] 
        else: 
            # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid 
            # failure. 
            best_answer = {"text": "", "score": 0.0} 
 
        predictions[example["id"]] = best_answer["text"] 
 
 
    return predictions 


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_json_file", help="Path to the json file", type=str)
    args = parser.parse_args()


    ### Parameters ###

    model = "distilbert"
    max_length = 384  # The maximum length of a feature (question and context)
    doc_stride = 128  # The authorized overlap between two part of the context 
                      # when splitting it is needed.
    batch_size = 32
    learning_rate = 0.0001
    num_warmup_steps = 0


    #### Load data ####

    path_to_json = args.path_to_json_file

    input_data = pd.read_json(path_to_json)
    
    print(f'length input dataset: {len(input_data["data"])}')
    
    data = flatten_input(input_data)

    print(f"length data after flattening: {len(data)}")

    ##### Load model ####

    qa_model = TFAutoModelForQuestionAnswering.from_pretrained(model)
    qa_model.load_weights(f'data/{model}.h5')

    #### Pre-processing ####

    tokenizer = AutoTokenizer.from_pretrained(model)

    dataset = Dataset.from_pandas(data)

    features = dataset.map(prepare_features,
                           batched=True,
                           remove_columns=dataset.column_names)

    data_collator = DefaultDataCollator(return_tensors="tf")

    dataset = features.to_tf_dataset(columns=["attention_mask", "input_ids"],
                                     shuffle=False,
                                     batch_size=batch_size,
                                     collate_fn=data_collator)

    #### Inference ####

    raw_predictions = qa_model.predict(dataset)

    #### Post-processing ####

    final_predictions = postprocess_qa_predictions(dataset,
                                                   features,                    raw_predictions)

    #### Saving the results ####

    with open(f'predictions/{model}_predictions.json', 'w') as json_file:
        json.dump(final_predictions, json_file)