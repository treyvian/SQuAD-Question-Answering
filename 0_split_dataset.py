"""First step loading the dataset and split it into training and validation"""

import json

# Read json file
path_to_json = "./data/training_set.json" 
with open(path_to_json, 'r') as f:
    input_data = json.load(f)

print(f'The input dataset is SQUAD version {input_data["version"]}')
print(f'lenght input dataset: {len(input_data["data"])}')


# Splitting the dataset into training and validation
split = 0.2 # Percentage for the validation
len_training = len(input_data['data']) * (1 - split)

data_training = []
data_validation = []

# Splitting as suggested based on the title
for i, article in enumerate(input_data['data']):
    # article is a dectionary with keys: title, paragraphs
    title = article['title'].strip()

    for paragraph in article['paragraphs']:
        # paragraph is a dectionary with keys: context, qas
        context = paragraph['context'].strip()

        for qa in paragraph["qas"]:
            # qa is a dectionary with keys: answers, question, id
            question = qa["question"].strip()
            id_ = qa["id"]

            answers_start = [answer["answer_start"] for answer in qa["answers"]]
            answers = [answer["text"].strip() for answer in qa["answers"]]

            if i <= len_training:
                data_training.append({'title': title,
                                      'context': context,
                                      'question': question,
                                      'id': id_,
                                      'answer': {
                                          'answers_start': answers_start,
                                          'text': answers
                                      }})
            else:
                data_validation.append({'title': title,
                                        'context': context,
                                        'question': question,
                                        'id': id_,
                                        'answer': {
                                          'answers_start': answers_start,
                                          'text': answers
                                        }})         

print(f"lenght training: {len(data_training)}")         
print(f"lenght validation: {len(data_validation)}") 

print("Writing files...")

with open('./data/train.json', 'w') as file:
    train_data = {'data': data_training}
    file.write(json.dumps(train_data))
    file.close()
              
with open('./data/validation.json', 'w') as file:
    val_data = {'data': data_validation}
    file.write(json.dumps(val_data))
    file.close()

print('Files written')


