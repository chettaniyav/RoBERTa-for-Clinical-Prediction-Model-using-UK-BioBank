from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from transformers.trainer_utils import EvalPrediction
import json
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, roc_curve, auc
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.optim as optim
import torch.nn as nn
from huggingface_hub import HfApi, HfFolder
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import TrainingArguments, Trainer, EvalPrediction
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from datetime import datetime


def create_sentence(row):
    current_year = datetime.now().year
    year_of_birth = row.get('Year of Birth')

    if year_of_birth is not None:
        age = current_year - int(year_of_birth)
    else:
        age = 'unknown age'

    sentence = (
        f"The patient is a {row.get('Sex', 'unknown gender')} born in {year_of_birth if year_of_birth is not None else 'unknown year'}"
        f" ({age} years old)."
    )

    # Basic physical attributes
    if row.get('Weight (kg)') is not None and row.get('Height (cm)') is not None:
        sentence += f" They weigh {row['Weight (kg)']} kg and are {row['Height (cm)']} cm tall."

    if row.get('Body Mass Index (BMI)') is not None:
        sentence += f" Their BMI is {row['Body Mass Index (BMI)']}."

    # Blood pressure
    if row.get('Systolic Blood Pressure (mmHg)') is not None and row.get('Diastolic Blood Pressure (mmHg)') is not None:
        sentence += (
            f" The systolic blood pressure is {row['Systolic Blood Pressure (mmHg)']} mmHg and "
            f"the diastolic blood pressure is {row['Diastolic Blood Pressure (mmHg)']} mmHg."
        )

    # Cholesterol and HIV antigens
    if row.get('Cholesterol (mg/dL)') is not None:
        sentence += f" Cholesterol level: {row['Cholesterol (mg/dL)']} mg/dL."

    if row.get('HIV-1 Gag Antigen') is not None:
        sentence += f" HIV-1 gag antigen result: {row['HIV-1 Gag Antigen']}."

    if row.get('HIV-1 Env Antigen') is not None:
        sentence += f" HIV-1 env antigen result: {row['HIV-1 Env Antigen']}."

    # Townsend Deprivation Index
    if row.get('Townsend Deprivation Index') is not None:
        sentence += f" Townsend deprivation index: {row['Townsend Deprivation Index']}."

    # Smoking and Alcohol
    if row.get('Current Tobacco Smoking Status') is not None:
        sentence += f" Current tobacco smoking status: {row['Current Tobacco Smoking Status']}."

    if row.get('Past Tobacco Smoking Status') is not None:
        sentence += f" Past tobacco smoking status: {row['Past Tobacco Smoking Status']}."

    if row.get('Alcohol Intake Frequency') is not None:
        sentence += f" Frequency of alcohol intake: {row['Alcohol Intake Frequency']}."

    # Diagnoses
    if row.get('Diabetes Diagnosis') is not None:
        sentence += f" Diagnosed with diabetes: {row['Diabetes Diagnosis']}."

    if row.get('Vascular/Heart Problem Diagnosis') is not None:
        sentence += f" Diagnosed with vascular/heart problems: {row['Vascular/Heart Problem Diagnosis']}."

    # Psychological Information
    if row.get('Seen GP for Anxiety/Depression') is not None:
        sentence += f" Seen a GP for anxiety or depression: {row['Seen GP for Anxiety/Depression']}."

    if row.get('Seen Psychiatrist for Anxiety/Depression') is not None:
        sentence += f" Seen a psychiatrist for anxiety or depression: {row['Seen Psychiatrist for Anxiety/Depression']}."

    # Ethnicity
    if row.get('Ethnic Background') is not None:
        sentence += f" The patient identifies as {row['Ethnic Background']}."

    return sentence


# Save your Hugging Face token
token = "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"  # Replace with your actual token
HfFolder.save_token(token)

# # Load the data
# data_path = '/mnt/bmh01-rds/Jenkins_HDS_dissertations/n68517cv/main/Data/Data_preprocessing/train_data.json'
# df = pd.read_json(data_path)
data_path = '/mnt/bmh01-rds/Jenkins_HDS_dissertations/n68517cv/data/X_train_new.csv'
# row = 100
df = pd.read_csv(data_path)
df.drop(columns=['Unnamed: 0'])
df.to_json('train_data.json', orient='records')
f = open("./train_data.json")
data_list = json.load(f)

# Val

# data_path = '/mnt/bmh01-rds/Jenkins_HDS_dissertations/n68517cv/main/Data/Data_preprocessing/train_data.json'
# df = pd.read_json(data_path)
data_path = '/mnt/bmh01-rds/Jenkins_HDS_dissertations/n68517cv/data/X_test_new.csv'
df = pd.read_csv(data_path)
df.drop(columns=['Unnamed: 0'])
df.to_json('test_data.json', orient='records')
f = open("./test_data.json")
test_data_list = json.load(f)


# data_list = json.dumps(data_list)
# Load Train and Test
train_texts = [create_sentence(row) for row in data_list]
test_texts = [create_sentence(row) for row in test_data_list]


# Load labels
# labels_path = '/mnt/bmh01-rds/Jenkins_HDS_dissertations/n68517cv/main/Data/Data_preprocessing/labels.txt'
# with open(labels_path, 'r') as f:
#     labels = f.readlines()

labels_path = '/mnt/bmh01-rds/Jenkins_HDS_dissertations/n68517cv/data/y_train_new.csv'
y_train = pd.read_csv(labels_path, header=None)
y_train.fillna('None of the above', inplace=True)
labels = y_train[1].tolist()

# Strip newline characters
# labels = [label.strip() for label in labels]
HA_labels = []
# # Function to predict if 'Heart Attack' is in the list


def predict_heart_attack(text):
    if 'Stroke' in text:
        return 'Stroke'
    else:
        return 'Not Stroke'


for i in labels:
    i = predict_heart_attack(i)
    HA_labels.append(i)
labels = HA_labels


# Create a mapping from text labels to integers
label_to_int = {label: idx for idx, label in enumerate(set(labels))}
int_labels = [label_to_int[label] for label in labels]

# Verify the mapping
print("Label mapping:", label_to_int)

# Define Dataset Class


class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Chettaniiay/HESRoberTA")
model = AutoModelForSequenceClassification.from_pretrained(
    "Chettaniiay/HESRoberTA", num_labels=len(label_to_int))

# Convert dataframe to list of texts
# texts = df.apply(lambda row: ' '.join(row.values.astype(str)), axis=1).tolist()

# Create the dataset
max_length = 128
train_dataset = TextClassificationDataset(
    train_texts, int_labels, tokenizer, max_length)
labels_path = '/mnt/bmh01-rds/Jenkins_HDS_dissertations/n68517cv/data/y_test_new.csv'
y_test = pd.read_csv(labels_path, header=None)
y_test.fillna('None of the above', inplace=True)
labels = y_test[1].tolist()

# Strip newline characters
# labels = [label.strip() for label in labels]
HA_labels = []
# # Function to predict if 'Heart Attack' is in the list


# def predict_heart_attack(text):
#     if 'Heart attack' in text:
#         return 'Heart attack'
#     else:
#         return 'Not Heart attack'


for i in labels:
    i = predict_heart_attack(i)
    HA_labels.append(i)
labels = HA_labels
label_to_int = {label: idx for idx, label in enumerate(set(labels))}
int_labels = [label_to_int[label] for label in labels]

val_dataset = TextClassificationDataset(
    test_texts, int_labels, tokenizer, max_length)
# # Create data loaders
# train_size = int(0.8 * len(dataset))
# val_size = len(dataset) - train_size
# train_dataset, val_dataset = torch.utils.data.random_split(
#     dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=1028, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=1028, shuffle=False)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=512,
    per_device_eval_batch_size=512,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    logging_dir='./logs',
    logging_steps=10,
)

# Define the metrics


def multi_label_metrics(predictions, labels, threshold=0.5):
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average='micro')
    accuracy = accuracy_score(y_true, y_pred)
    metrics = {'f1': f1_micro_average,
               'roc_auc': roc_auc, 'accuracy': accuracy}
    return metrics

# Define the metrics for multiclass classification


def adjust_size(arr, target_size=100):
    """Truncate or pad the array to make it of size target_size."""
    current_size = len(arr)
    if current_size > target_size:
        return arr[:target_size]  # Truncate

    # elif current_size < target_size:
    #     # Pad with zeros
    #     return np.pad(arr, (0, target_size - current_size), 'constant')
    # return arr


def bootstrap_confidence_interval(metric_function, labels, preds, n_bootstraps=1000, ci=95):
    rng = np.random.RandomState(42)  # Seed for reproducibility
    bootstrapped_scores = []

    for _ in range(n_bootstraps):
        # Resample with replacement
        indices = rng.randint(0, len(labels), len(labels))
        # Skip if there's only one class in the resample
        if len(np.unique(labels[indices])) < 2:
            continue
        score = metric_function(labels[indices], preds[indices])
        bootstrapped_scores.append(score)

    # Calculate the confidence interval
    sorted_scores = np.sort(bootstrapped_scores)
    lower_bound = np.percentile(sorted_scores, (100 - ci) / 2)
    upper_bound = np.percentile(sorted_scores, 100 - (100 - ci) / 2)

    return lower_bound, upper_bound


def compute_metrics(predictions, threshold=0.5):
    preds = predictions.predictions.argmax(-1)
    labels = predictions.label_ids

    # Calculate precision, recall, F1-score, and accuracy
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted')
    accuracy = accuracy_score(labels, preds)

    # Calculate confidence intervals using bootstrapping
    ci_lower_ac, ci_upper_ac = bootstrap_confidence_interval(
        accuracy_score, labels, preds)
    ci_lower_pre, ci_upper_pre = bootstrap_confidence_interval(
        lambda y_true, y_pred: precision_score(y_true, y_pred, average='weighted'), labels, preds)
    ci_lower_re, ci_upper_re = bootstrap_confidence_interval(
        lambda y_true, y_pred: recall_score(y_true, y_pred, average='weighted'), labels, preds)
    ci_lower_f1, ci_upper_f1 = bootstrap_confidence_interval(
        lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted'), labels, preds)

    # Return the metrics as a dictionary
    metrics = {
        'accuracy score': {'accuracy': accuracy, 'CI': f"95% Confidence Interval: [{ci_lower_ac:.4f}, {ci_upper_ac:.4f}]"},
        'precision score': {'precision': precision, 'CI': f"95% Confidence Interval: [{ci_lower_pre:.4f}, {ci_upper_pre:.4f}]"},
        'recall score': {'recall': recall, 'CI': f"95% Confidence Interval: [{ci_lower_re:.4f}, {ci_upper_re:.4f}]"},
        'f1 score': {'f1': f1, 'CI': f"95% Confidence Interval: [{ci_lower_f1:.4f}, {ci_upper_f1:.4f}]"},
    }

    return metrics


# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()

# Print the evaluation results
print(results)

model_save_path = "./fine_tune"
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

# Get predictions
predictions = trainer.predict(val_dataset)
val_results = compute_metrics(predictions)
print('val results:', val_results)
# Define the file path where you want to save the results
file_path = 'val_results_HD.txt'

# Open the file in write mode ('w') and write the results
with open(file_path, 'w') as file:
    # Write a header
    file.write('Validation Results:\n')

    # Write each result key-value pair
    for key, value in val_results.items():
        file.write(f'{key}: {value}\n')

print(f'Validation results have been written to {file_path}')
# Extract loss and epoch information
epochs = []
train_loss = []
eval_loss = []
training_logs = trainer.state.log_history
# Separate the logs based on the type of loss recorded
for log in training_logs:
    if 'epoch' in log:
        epoch = log['epoch']
        if 'loss' in log:
            train_loss.append((epoch, log['loss']))
        if 'eval_loss' in log:
            eval_loss.append((epoch, log['eval_loss']))

# Sort the loss logs based on epoch
train_loss = sorted(train_loss, key=lambda x: x[0])
eval_loss = sorted(eval_loss, key=lambda x: x[0])

# Extract epochs and loss values
epochs_train, train_loss = zip(*train_loss)
epochs_eval, eval_loss = zip(*eval_loss)

# Convert to pandas DataFrame for easier handling and smoothing
df_train = pd.DataFrame({'epoch': epochs_train, 'train_loss': train_loss})
df_eval = pd.DataFrame({'epoch': epochs_eval, 'eval_loss': eval_loss})

# Adding initial epoch for validation loss if necessary
if df_eval['epoch'].iloc[0] != df_train['epoch'].iloc[0]:
    df_eval = pd.concat([pd.DataFrame({'epoch': [df_train['epoch'].iloc[0]], 'eval_loss': [
                        None]}), df_eval]).reset_index(drop=True)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(df_train['epoch'], df_train['train_loss'],
         label='Training Loss', marker='o', linestyle='-', color='b')
plt.plot(df_eval['epoch'], df_eval['eval_loss'],
         label='Validation Loss', marker='o', linestyle='-', color='orange')

# Adding titles and labels
plt.title('Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')

# Adding grid
plt.grid(True, linestyle='--', alpha=0.7)

# Adding legend
plt.legend()

# Save the plot as PNG
plt.savefig('./loss_vs_epochs.png')
# plt.show()


####### ROC ###########


# Compute ROC curve and AUC
fpr, tpr, _ = roc_curve(predictions.label_ids, predictions.predictions[:, 1])
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
# plt.show()
plt.savefig('./roc.png')
