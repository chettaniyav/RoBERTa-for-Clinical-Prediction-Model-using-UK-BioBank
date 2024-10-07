import seaborn as sns
import numpy as np
import re
import torch.nn.functional as F
import shap
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer
import torch
import matplotlib.pyplot as plt
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

    # Additional info
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


# Load test data and labels
labels_path = "/mnt/bmh01-rds/Jenkins_HDS_dissertations/n68517cv/data/y_train_new.csv"
y_test = pd.read_csv(labels_path, header=None)
y_test.fillna('None of the above', inplace=True)
labels = y_test[1].tolist()

f = open(f'/mnt/bmh01-rds/Jenkins_HDS_dissertations/n68517cv/main/SUBERT/model/Fine-Tune/Test/St/train_data.json')
test_data_list = json.load(f)
test_texts = [create_sentence(row) for row in test_data_list]

# Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(
    "/mnt/bmh01-rds/Jenkins_HDS_dissertations/n68517cv/main/SUBERT/model/Fine-Tune/Test/HD/fine_tune")
model = AutoModelForSequenceClassification.from_pretrained(
    "/mnt/bmh01-rds/Jenkins_HDS_dissertations/n68517cv/main/SUBERT/model/Fine-Tune/Test/HD/fine_tune")

conditions = ["Heart attack", "Not Heart attack"]
text = 'The patient is a Male born in 1943 of Age 81 years old. They weigh 87.7 kg and are 174.3401963959 cm tall. Their BMI is 28.9668. The systolic blood pressure is 136.0 mmHg and the diastolic blood pressure is 75.0 mmHg. Cholesterol level: 4.272 mg/dL. HIV-1 gag antigen result: 125.2130706137. HIV-1 env antigen result: 48.7560469214. Townsend deprivation index: -4.7565. Current tobacco smoking status: No. Past tobacco smoking status: Smoked on most or all days. Frequency of alcohol intake: Daily or almost daily. Diagnosed with diabetes: No. Seen a GP for anxiety or depression: No. Seen a psychiatrist for anxiety or depression: Yes. The patient identifies as British.'


def predictor(texts):
    inputs = tokenizer(texts, return_tensors="pt",
                       padding=True, truncation=True)
    outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=1).detach().numpy()
    return probs
# import shap

# # Load test data and labels
# labels_path = "/mnt/bmh01-rds/Jenkins_HDS_dissertations/n68517cv/data/y_test_new.csv"
# y_test = pd.read_csv(labels_path, header=None)
# y_test.fillna('None of the above', inplace=True)
# labels = y_test[1].tolist()

# f = open(f'/mnt/bmh01-rds/Jenkins_HDS_dissertations/n68517cv/main/SUBERT/model/Fine-Tune/Test/St/test_data.json')
# test_data_list = json.load(f)
# test_texts = [create_sentence(row) for row in test_data_list]


# Initialize the tokenizer and model
model_name = "Chettaniiay/HESRoberTA-HA"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

conditions = ["Heart attack", "Not a Heart attack"]
explainer = LimeTextExplainer(
    class_names=["Not Heart Disease", "Heart Disease"])
# Get LIME explanation for the text
explanation = explainer.explain_instance(text, predictor, num_features=20)

# Extract the token importance scores from the LIME explanation
tokens = []
importance_scores = []

for word, score in explanation.as_list():
    tokens.append(word)
    importance_scores.append(score)

# Convert importance scores to a numpy array and normalize
importance_scores = np.array(importance_scores)
importance_scores_normalized = (importance_scores - importance_scores.min()) / (
    importance_scores.max() - importance_scores.min())

# Plot the heatmap
plt.figure(figsize=(12, 1))
sns.heatmap([importance_scores_normalized], annot=[tokens],
            cmap='coolwarm', cbar=False, xticklabels=False)
plt.title("Token Importance Heatmap")
plt.show()

# from transformers import AutoTokenizer, AutoModelForSequenceClassification

# filename_model = 'ProsusAI/finbert'
# tokenizer = AutoTokenizer.from_pretrained(filename_model)
# model = AutoModelForSequenceClassification.from_pretrained(filename_model)
# class_names = ['positive', 'negative', 'neutral']


# def predictor(texts):
#     outputs = model(**tokenizer(texts, return_tensors="pt", padding=True))
#     tensor_logits = outputs[0]
#     probas = F.softmax(tensor_logits).detach().numpy()
#     return probas


# text = 'The patient is a Male born in 1943 of Age 81 years old. They weigh 87.7 kg and are 174.3401963959 cm tall. Their BMI is 28.9668. The systolic blood pressure is 136.0 mmHg and the diastolic blood pressure is 75.0 mmHg. Cholesterol level: 4.272 mg/dL. HIV-1 gag antigen result: 125.2130706137. HIV-1 env antigen result: 48.7560469214. Townsend deprivation index: -4.7565. Current tobacco smoking status: No. Past tobacco smoking status: Smoked on most or all days. Frequency of alcohol intake: Daily or almost daily. Diagnosed with diabetes: No. Seen a GP for anxiety or depression: No. Seen a psychiatrist for anxiety or depression: Yes. The patient identifies as British.'
# # text = test_texts[1]
# import re
# text = re.sub(r'\.\d+', '', text)

# print(tokenizer(text, return_tensors='pt', padding=True))

# explainer = LimeTextExplainer(class_names=conditions)
# exp = explainer.explain_instance(
#     text, predictor, num_features=20, num_samples=2000)
# exp.show_in_notebook(text=text)


# # from transformers import AutoTokenizer, AutoModelForSequenceClassification

# # filename_model = 'ProsusAI/finbert'
# # tokenizer = AutoTokenizer.from_pretrained(filename_model)
# # model = AutoModelForSequenceClassification.from_pretrained(filename_model)
# # class_names = ['positive', 'negative', 'neutral']


# def predictor(texts):
#     outputs = model(**tokenizer(texts, return_tensors="pt", padding=True))
#     tensor_logits = outputs[0]
#     probas = F.softmax(tensor_logits).detach().numpy()
#     return probas


# # text = 'I love you but I hate you more'
# index = labels.index('Heart attack')
# text = test_texts[index]

# print(text, labels[index])
# # for i in labels:
# #     if i in conditions:
# #         print(i.index())
# # for i in labels:
# #     if i in labels:

# # print(tokenizer(text, return_tensors='pt', padding=True))
# # text = "The temperature is 25.0 degrees, and the result is 100.0."
# # Remove ".0" from the text
# text = re.sub(r'\.\d+', '', text)
# explainer = LimeTextExplainer(class_names=conditions)
# exp = explainer.explain_instance(
#     text, predictor, num_features=20, num_samples=2000)
# exp.show_in_notebook(text=text)
# exp.save_to_file('./lime_explanation.html')

# shap.save_html('./shap_explanation.html', exp)

# # Define the prediction function


# # def predict_fn(texts):
# #     inputs = tokenizer(texts, padding=True,
# #                        truncation=True, return_tensors="pt")
# #     with torch.no_grad():
# #         outputs = model(**inputs)
# #         probs = torch.softmax(outputs.logits, dim=-1)
# #     return probs.numpy()


# # # Initialize the LimeTextExplainer
# # # Replace with actual label names
# # explainer = LimeTextExplainer(
# #     class_names=["Heart attack", "Not Heart attack"])

# # # Text to explain
# # text = test_texts[1]

# # # Get explanations
# # explanation = explainer.explain_instance(text, predict_fn, num_features=20)

# # # Extract the data to plot
# # exp = explanation.as_list()

# # # Plot the explanation using matplotlib
# # fig, ax = plt.subplots(figsize=(10, 6))

# # labels, scores = zip(*exp)
# # y_pos = range(len(labels))

# # ax.barh(y_pos, scores, align='center', color=[
# #         'green' if x > 0 else 'red' for x in scores])
# # ax.set_yticks(y_pos)
# # ax.set_yticklabels(labels)
# # ax.invert_yaxis()  # Invert y-axis so that the most important feature is on top
# # ax.set_xlabel('Importance Score')
# # ax.set_title(f'Explanation for: "{text}"')

# # # Save the plot as an image file
# # # plt.savefig('lime_explanation.png')

# # # Optionally, display the plot in a non-notebook environment
# # plt.show()
# # Define a wrapper function to make predictions


# # def predict_fn(input_ids):
# #     inputs = {'input_ids': torch.tensor(input_ids)}
# #     outputs = model(**inputs)
# #     return outputs.logits.detach().numpy()


# # # Use the KernelExplainer
# # # background_data: representative data
# # explainer = shap.KernelExplainer(predict_fn, data=test_texts)
# # shap_values = explainer.shap_values(test_texts)

# # shap.force_plot(explainer.expected_value,
# #                 shap_values[0], feature_names=tokenizer.convert_ids_to_tokens(test_texts))
