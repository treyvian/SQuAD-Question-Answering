import argparse
import json
import pandas as pd
from transformers import AutoTokenizer
from datasets import Dataset


def flatten_input(input_data):
    """Flattens the input data"""
    
    data = []

    for i, article in enumerate(input_data['data']):
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
    return data


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_json_file", help="Path to the json file", type=str)
    args = parser.parse_args()


    ### Parameters ###

    model = "distilbert-base-uncased"
    # The maximum length of a feature (question and context)
    max_length = 384
    # The authorized overlap between two part of the context when splitting it is needed.
    doc_stride = 128


    #### Load data ####

    path_to_json = args.path_to_json_file

    with open(path_to_json, 'r') as f:
        input_data = json.load(f)
    
    print(f'length input dataset: {len(input_data["data"])}')
    
    data = flatten_input(input_data)

    print(f"length data after flattening: {len(data)}")

    df_data = pd.DataFrame(data)

    #### Pre-processing ####

    tokenizer = AutoTokenizer.from_pretrained(model)

    dataset = Dataset.from_pandas(df_data)

    # Load model

