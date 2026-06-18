import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer


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


labels_path = "./data/y_train.csv"
y_test = pd.read_csv(labels_path, header=None)
y_test.fillna('None of the above', inplace=True)
labels = y_test[1].tolist()

with open('./data/train_data.json') as f:
    test_data_list = json.load(f)
test_texts = [create_sentence(row) for row in test_data_list]

model_name = "YOUR_HF_USERNAME/YOUR_FINETUNED_MODEL"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

text = (
    'The patient is a Male born in 1950 (75 years old). They weigh 80.0 kg and are 175.0 cm tall. '
    'Their BMI is 26.1. The systolic blood pressure is 130.0 mmHg and the diastolic blood pressure '
    'is 80.0 mmHg. Cholesterol level: 5.0 mg/dL. Townsend deprivation index: -2.0. Current tobacco '
    'smoking status: No. Past tobacco smoking status: Never smoked. Frequency of alcohol intake: '
    'Once or twice a week. Diagnosed with diabetes: No. Seen a GP for anxiety or depression: No. '
    'Seen a psychiatrist for anxiety or depression: No. The patient identifies as British.'
)


def predictor(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=1).detach().numpy()
    return probs


explainer = LimeTextExplainer(class_names=["Not Heart Disease", "Heart Disease"])
explanation = explainer.explain_instance(text, predictor, num_features=20)

tokens = []
importance_scores = []
for word, score in explanation.as_list():
    tokens.append(word)
    importance_scores.append(score)

importance_scores = np.array(importance_scores)
importance_scores_normalized = (importance_scores - importance_scores.min()) / (
    importance_scores.max() - importance_scores.min()
)

plt.figure(figsize=(12, 1))
sns.heatmap([importance_scores_normalized], annot=[tokens],
            cmap='coolwarm', cbar=False, xticklabels=False)
plt.title("Token Importance Heatmap")
plt.show()
