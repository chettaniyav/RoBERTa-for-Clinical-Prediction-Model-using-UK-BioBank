# import tensorflow as tf
# from transformers import BertTokenizer, TFBertForMaskedLM, DataCollatorForLanguageModeling, create_optimizer
# from datasets import Dataset

# # Load and preprocess data


# def load_text_data(file_path):
#     with open(file_path, 'r') as file:
#         lines = file.readlines()
#     # Assuming each line in the file is a separate training example
#     data = [{'text': line.strip()} for line in lines]
#     return data


# # Convert data to Hugging Face dataset
# file_path = '/mnt/bmh01-rds/Jenkins_HDS_dissertations/n68517cv/main/Data/Data_preprocessing/train.txt'
# data = load_text_data(file_path)
# dataset = Dataset.from_list(data)
# # Tokenize data
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# def tokenize_function(example):
#     return tokenizer(example['text'], padding='max_length', truncation=True, max_length=128)


# tokenized_dataset = dataset.map(tokenize_function, batched=True)

# # Data collator for MLM
# data_collator = DataCollatorForLanguageModeling(
#     tokenizer=tokenizer,
#     mlm=True,
#     mlm_probability=0.15
# )

# # Convert the dataset to TensorFlow format


# def gen():
#     for ex in tokenized_dataset:
#         yield ({'input_ids': ex['input_ids'], 'attention_mask': ex['attention_mask']}, ex['input_ids'])


# tf_dataset = tf.data.Dataset.from_generator(
#     gen,
#     ({'input_ids': tf.int32, 'attention_mask': tf.int32}, tf.int32),
#     ({'input_ids': tf.TensorShape([128]), 'attention_mask': tf.TensorShape(
#         [128])}, tf.TensorShape([128]))
# )

# tf_dataset = tf_dataset.shuffle(buffer_size=1024).batch(8)

# # Initialize the BERT model
# model = TFBertForMaskedLM.from_pretrained('bert-base-uncased')

# # Calculate the number of training steps
# num_train_steps = tf.data.experimental.cardinality(
#     tf_dataset).numpy() * 3  # 3 epochs

# # Compile the model
# optimizer, schedule = create_optimizer(
#     init_lr=2e-5,
#     num_train_steps=num_train_steps,
#     num_warmup_steps=int(0.1 * num_train_steps)
# )

# model.compile(optimizer=optimizer)

# # Train the model
# model.fit(tf_dataset, epochs=3)

# # Save the model
# save_path = './'
# model.save_pretrained(save_path)
# tokenizer.save_pretrained(save_path)

import json
from huggingface_hub import login
from huggingface_hub import HfApi, HfFolder, Repository, create_repo, upload_folder
from transformers import BertTokenizerFast, BertForMaskedLM
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizerFast, BertForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import matplotlib.pyplot as plt
# Load model directly
from transformers import AutoTokenizer, AutoModelForMaskedLM
from datetime import datetime

tokenizer = AutoTokenizer.from_pretrained("Chettaniiay/RoHBERTA")
model = AutoModelForMaskedLM.from_pretrained("Chettaniiay/RoHBERTA")
# Load the data
# Update this path if necessary
data_path = '/mnt/bmh01-rds/Jenkins_HDS_dissertations/n68517cv/data/MLM_data.json'

# file_path = '/mnt/bmh01-rds/Jenkins_HDS_dissertations/n68517cv/main/Data/Data_preprocessing/train.txt'
# df = pd.read_json(data_path)

# # Fill or handle NaN values
# df = df.fillna('')

# # Display the first few rows to verify
# print(df.head())
print(10*"Init"*10)
f = open(data_path)
data = json.load(f)

print(10*"Loading Json"*10)


def create_sentence(row):
    current_year = datetime.now().year
    year_of_birth = row.get('YearOfBirth')

    if year_of_birth is not None:
        age = current_year - int(year_of_birth)
    else:
        age = 'unknown age'

    sentence = (
        f"The patient is a {row.get('Sex', 'unknown gender')} born in {year_of_birth if year_of_birth is not None else 'unknown year'}"
        f" ({age} years old)."
    )

    if row.get('Weight') is not None and row.get('Height') is not None:
        sentence += f" They weigh {row['Weight']} kg and are {row['Height']} cm tall."

    if row.get('BodyMassIndex(Bmi)') is not None:
        sentence += f" Their BMI is {row['BodyMassIndex(Bmi)']}."

    if row.get('SystolBloodPressAutomRead') is not None and row.get('DiastolBloodPressAutomRead') is not None:
        sentence += (
            f" The systolic blood pressure is {row['SystolBloodPressAutomRead']} mmHg and "
            f"the diastolic blood pressure is {row['DiastolBloodPressAutomRead']} mmHg."
        )

    if row.get('EthnicBackground') is not None:
        sentence += f" The patient identifies as {row['EthnicBackground']}."

    if row.get('MoodSwing') is not None:
        sentence += f" They have experienced mood swings: {row['MoodSwing']}."

    if row.get('AvTotHouseholdIncomTax') is not None:
        sentence += f" The average total household income is {row['AvTotHouseholdIncomTax']}."

    if row.get('NervousFeel') is not None:
        sentence += f" The patient reports feeling nervous: {row['NervousFeel']}."

    if row.get('WorrierAnxiousFeel') is not None:
        sentence += f" They have expressed feelings of anxiety: {row['WorrierAnxiousFeel']}."

    if row.get('UsualWalkPace') is not None:
        sentence += f" The usual walking pace is described as {row['UsualWalkPace']}."

    if row.get('CurrentTobaccoSmoke') is not None:
        sentence += f" Current tobacco smoking status: {row['CurrentTobaccoSmoke']}."

    if row.get('PastTobaccoSmoke') is not None:
        sentence += f" Past tobacco smoking status: {row['PastTobaccoSmoke']}."

    if row.get('AlcoholIntakFrequenc') is not None:
        sentence += f" Frequency of alcohol intake: {row['AlcoholIntakFrequenc']}."

    if row.get('DiabetDiagnosDoct') is not None:
        sentence += f" Diagnosed with diabetes: {row['DiabetDiagnosDoct']}."

    if row.get('Irrit') is not None:
        sentence += f" They have experienced irritability: {row['Irrit']}."

    if row.get('FrequDepressMoodLast2Week') is not None:
        sentence += f" Frequency of depressed mood in the last 2 weeks: {row['FrequDepressMoodLast2Week']}."

    if row.get('FrequUnenthusiasmDisinterestLast2Week') is not None:
        sentence += f" Frequency of unenthusiasm/disinterest in the last 2 weeks: {row['FrequUnenthusiasmDisinterestLast2Week']}."

    if row.get('SleepDurat') is not None:
        sentence += f" The average sleep duration is {row['SleepDurat']} hours."

    if row.get('SleeplessInsomnia') is not None:
        sentence += f" They have experienced sleeplessness/insomnia: {row['SleeplessInsomnia']}."

    if row.get('Vascular/heartProblemDiagnosDoct') is not None:
        if row['Vascular/heartProblemDiagnosDoct'] == 'None of the above':
            sentence += f" Diagnosed with vascular/heart problems: No"
        else:
            sentence += f" Diagnosed with vascular/heart problems: {row['Vascular/heartProblemDiagnosDoct']}."

    for i in range(9):  # Loop through diagnosis_icd_0 to diagnosis_icd_8
        key = f'diagnosis_icd_{i}'
        if row.get(key) is not None:
            sentence += f" Diagnosed with {key}: {row[key]}."

    return sentence


# Example usage with a list of dictionaries
data_list = data

# Generating sentences for all entries
texts = [create_sentence(row) for row in data_list]
# Tokenize the texts
# tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
# texts = [' '.join(str(value) for value in item.values()) for item in data]

encodings = tokenizer(texts, truncation=True, padding=True,
                      max_length=128, return_tensors='pt')

print(10*"Token generated"*10)
# Prepare input data for PyTorch
inputs = encodings['input_ids']
attention_mask = encodings['attention_mask']

# Create a dataset class


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, attention_mask):
        self.inputs = inputs
        self.attention_mask = attention_mask

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        item = {
            'input_ids': self.inputs[idx],
            'attention_mask': self.attention_mask[idx]
        }
        return item


dataset = TextDataset(inputs, attention_mask)

# Load the BERT model
# model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.resize_token_embeddings(len(tokenizer))

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

print(10*"Model Init"*10)
# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=32,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=200,
    report_to=None,
    learning_rate=5e-5,  # Alpha
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-8,
    weight_decay=0.01,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)


# Check if multiple GPUs are available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = torch.nn.DataParallel(model)

# Move model to the appropriate device
model.to(device)

print(10*"Model Training Now"*10)
# Train the model
trainer.train()
print(10*"Model Trained and Saving Now"*10)

# Save the model and tokenizer
model_save_path = "./mlm"
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

print(10*"Generating Loss and Epochs plot"*10)

# Plot
training_logs = trainer.state.log_history
train_losses = [log['loss'] for log in training_logs if 'loss' in log]

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Training Loss Over Steps MLM')
plt.legend()
# plt.show()
plt.savefig(f"training_loss.png")  # Save the plot


print(10*"Pushing Model to HF "*10)


# Authenticate to Hugging Face

# Login to your Hugging Face account
# Replace 'YOUR_HF_TOKEN' with your actual Hugging Face token
login(token="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")

# # Paths
model_save_path = "path/to/save/model"
# Replace with your Hugging Face username and desired model name
hf_repo_name = "Chettaniiay/HESRoberTA"

# Load your trained model and tokenizer
tokenizer = BertTokenizerFast.from_pretrained(model_save_path)
model = BertForMaskedLM.from_pretrained(model_save_path)

# Save the model and tokenizer to the local directory
tokenizer.save_pretrained(model_save_path)
model.save_pretrained(model_save_path)

# Create a new repo on the Hugging Face Hub (if it doesn't already exist)
create_repo(hf_repo_name, exist_ok=True)

# Upload the model to the Hub
upload_folder(
    repo_id=hf_repo_name,
    folder_path=model_save_path,
    commit_message="Initial commit of the trained model",
)

print(f"Model pushed to https://huggingface.co/{hf_repo_name}")
