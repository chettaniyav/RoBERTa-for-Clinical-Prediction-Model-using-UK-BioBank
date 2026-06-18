import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from huggingface_hub import HfFolder
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

token = os.getenv("HF_TOKEN")
HfFolder.save_token(token)


def create_sentence(row):
    current_year = datetime.now().year
    year_of_birth = row.get('Year of Birth')
    age = current_year - int(year_of_birth) if year_of_birth is not None else 'unknown age'

    sentence = (
        f"The patient is a {row.get('Sex', 'unknown gender')} born in "
        f"{year_of_birth if year_of_birth is not None else 'unknown year'} ({age} years old)."
    )

    if row.get('Weight (kg)') is not None and row.get('Height (cm)') is not None:
        sentence += f" They weigh {row['Weight (kg)']} kg and are {row['Height (cm)']} cm tall."

    if row.get('Body Mass Index (BMI)') is not None:
        sentence += f" Their BMI is {row['Body Mass Index (BMI)']}."

    if row.get('Systolic Blood Pressure (mmHg)') is not None and row.get('Diastolic Blood Pressure (mmHg)') is not None:
        sentence += (
            f" The systolic blood pressure is {row['Systolic Blood Pressure (mmHg)']} mmHg and "
            f"the diastolic blood pressure is {row['Diastolic Blood Pressure (mmHg)']} mmHg."
        )

    if row.get('Cholesterol (mg/dL)') is not None:
        sentence += f" Cholesterol level: {row['Cholesterol (mg/dL)']} mg/dL."

    if row.get('HIV-1 Gag Antigen') is not None:
        sentence += f" HIV-1 gag antigen result: {row['HIV-1 Gag Antigen']}."

    if row.get('HIV-1 Env Antigen') is not None:
        sentence += f" HIV-1 env antigen result: {row['HIV-1 Env Antigen']}."

    if row.get('Townsend Deprivation Index') is not None:
        sentence += f" Townsend deprivation index: {row['Townsend Deprivation Index']}."

    if row.get('Current Tobacco Smoking Status') is not None:
        sentence += f" Current tobacco smoking status: {row['Current Tobacco Smoking Status']}."

    if row.get('Past Tobacco Smoking Status') is not None:
        sentence += f" Past tobacco smoking status: {row['Past Tobacco Smoking Status']}."

    if row.get('Alcohol Intake Frequency') is not None:
        sentence += f" Frequency of alcohol intake: {row['Alcohol Intake Frequency']}."

    if row.get('Diabetes Diagnosis') is not None:
        sentence += f" Diagnosed with diabetes: {row['Diabetes Diagnosis']}."

    if row.get('Vascular/Heart Problem Diagnosis') is not None:
        sentence += f" Diagnosed with vascular/heart problems: {row['Vascular/Heart Problem Diagnosis']}."

    if row.get('Seen GP for Anxiety/Depression') is not None:
        sentence += f" Seen a GP for anxiety or depression: {row['Seen GP for Anxiety/Depression']}."

    if row.get('Seen Psychiatrist for Anxiety/Depression') is not None:
        sentence += f" Seen a psychiatrist for anxiety or depression: {row['Seen Psychiatrist for Anxiety/Depression']}."

    if row.get('Ethnic Background') is not None:
        sentence += f" The patient identifies as {row['Ethnic Background']}."

    return sentence


data_path = './data/X_train.csv'
df = pd.read_csv(data_path)
df.drop(columns=['Unnamed: 0'])
df.to_json('train_data.json', orient='records')
with open("./train_data.json") as f:
    data_list = json.load(f)

data_path = './data/X_test.csv'
df = pd.read_csv(data_path)
df.drop(columns=['Unnamed: 0'])
df.to_json('test_data.json', orient='records')
with open("./test_data.json") as f:
    test_data_list = json.load(f)

train_texts = [create_sentence(row) for row in data_list]
test_texts = [create_sentence(row) for row in test_data_list]

labels_path = './data/y_train.csv'
y_train = pd.read_csv(labels_path, header=None)
y_train.fillna('None of the above', inplace=True)
labels = y_train[1].tolist()


def predict_condition(text):
    return 'Heart attack' if 'Heart attack' in text else 'Not Heart attack'


labels = [predict_condition(label) for label in labels]
label_to_int = {label: idx for idx, label in enumerate(set(labels))}
int_labels = [label_to_int[label] for label in labels]
print("Label mapping:", label_to_int)


class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer.encode_plus(
            self.texts[idx],
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
            'labels': torch.tensor(self.labels[idx], dtype=torch.long),
        }


tokenizer = AutoTokenizer.from_pretrained("YOUR_HF_USERNAME/YOUR_MLM_MODEL")
model = AutoModelForSequenceClassification.from_pretrained(
    "YOUR_HF_USERNAME/YOUR_MLM_MODEL", num_labels=len(label_to_int))

max_length = 128
train_dataset = TextClassificationDataset(train_texts, int_labels, tokenizer, max_length)

labels_path = './data/y_test.csv'
y_test = pd.read_csv(labels_path, header=None)
y_test.fillna('None of the above', inplace=True)
labels = y_test[1].tolist()
labels = [predict_condition(label) for label in labels]
label_to_int = {label: idx for idx, label in enumerate(set(labels))}
int_labels = [label_to_int[label] for label in labels]

val_dataset = TextClassificationDataset(test_texts, int_labels, tokenizer, max_length)


def bootstrap_confidence_interval(metric_function, labels, preds, n_bootstraps=1000, ci=95):
    rng = np.random.RandomState(42)
    bootstrapped_scores = []
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(labels), len(labels))
        if len(np.unique(labels[indices])) < 2:
            continue
        bootstrapped_scores.append(metric_function(labels[indices], preds[indices]))
    sorted_scores = np.sort(bootstrapped_scores)
    lower = np.percentile(sorted_scores, (100 - ci) / 2)
    upper = np.percentile(sorted_scores, 100 - (100 - ci) / 2)
    return lower, upper


def compute_metrics(predictions, threshold=0.5):
    preds = predictions.predictions.argmax(-1)
    labels = predictions.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    accuracy = accuracy_score(labels, preds)
    ci_lower_ac, ci_upper_ac = bootstrap_confidence_interval(accuracy_score, labels, preds)
    ci_lower_pre, ci_upper_pre = bootstrap_confidence_interval(
        lambda y_true, y_pred: precision_score(y_true, y_pred, average='weighted'), labels, preds)
    ci_lower_re, ci_upper_re = bootstrap_confidence_interval(
        lambda y_true, y_pred: recall_score(y_true, y_pred, average='weighted'), labels, preds)
    ci_lower_f1, ci_upper_f1 = bootstrap_confidence_interval(
        lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted'), labels, preds)
    return {
        'accuracy score': {'accuracy': accuracy, 'CI': f"95% CI: [{ci_lower_ac:.4f}, {ci_upper_ac:.4f}]"},
        'precision score': {'precision': precision, 'CI': f"95% CI: [{ci_lower_pre:.4f}, {ci_upper_pre:.4f}]"},
        'recall score': {'recall': recall, 'CI': f"95% CI: [{ci_lower_re:.4f}, {ci_upper_re:.4f}]"},
        'f1 score': {'f1': f1, 'CI': f"95% CI: [{ci_lower_f1:.4f}, {ci_upper_f1:.4f}]"},
    }


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

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

results = trainer.evaluate()
print(results)

model_save_path = "./fine_tune"
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

predictions = trainer.predict(val_dataset)
val_results = compute_metrics(predictions)
print('Validation results:', val_results)

with open('val_results_HD.txt', 'w') as f:
    f.write('Validation Results:\n')
    for key, value in val_results.items():
        f.write(f'{key}: {value}\n')

training_logs = trainer.state.log_history
train_loss = sorted(
    [(log['epoch'], log['loss']) for log in training_logs if 'epoch' in log and 'loss' in log],
    key=lambda x: x[0],
)
eval_loss = sorted(
    [(log['epoch'], log['eval_loss']) for log in training_logs if 'epoch' in log and 'eval_loss' in log],
    key=lambda x: x[0],
)

epochs_train, train_loss_vals = zip(*train_loss)
epochs_eval, eval_loss_vals = zip(*eval_loss)

df_train = pd.DataFrame({'epoch': epochs_train, 'train_loss': train_loss_vals})
df_eval = pd.DataFrame({'epoch': epochs_eval, 'eval_loss': eval_loss_vals})

if df_eval['epoch'].iloc[0] != df_train['epoch'].iloc[0]:
    df_eval = pd.concat([
        pd.DataFrame({'epoch': [df_train['epoch'].iloc[0]], 'eval_loss': [None]}),
        df_eval,
    ]).reset_index(drop=True)

plt.figure(figsize=(10, 6))
plt.plot(df_train['epoch'], df_train['train_loss'], label='Training Loss', marker='o', color='b')
plt.plot(df_eval['epoch'], df_eval['eval_loss'], label='Validation Loss', marker='o', color='orange')
plt.title('Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.savefig('./loss_vs_epochs.png')

fpr, tpr, _ = roc_curve(predictions.label_ids, predictions.predictions[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('./roc.png')

predictions = trainer.predict(val_dataset)
preds = predictions.predictions.argmax(-1)
labels = predictions.label_ids
class_report = classification_report(labels, preds, target_names=label_to_int.keys())
print('Classification Report:\n', class_report)

with open('classification_report.txt', 'w') as f:
    f.write('Classification Report:\n')
    f.write(class_report)
