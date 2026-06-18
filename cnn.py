import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
from huggingface_hub import HfApi, HfFolder
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import (
    Input, Embedding, LSTM, Dense, Dropout, Bidirectional,
)

token = os.getenv("HF_TOKEN")
HfFolder.save_token(token)


def create_sentence(row):
    current_year = datetime.now().year
    year_of_birth = row.get('YearOfBirth')
    age = current_year - int(year_of_birth) if year_of_birth is not None else 'unknown age'

    sentence = (
        f"The patient is a {row.get('Sex', 'unknown gender')} born in "
        f"{year_of_birth if year_of_birth is not None else 'unknown year'} ({age} years old)."
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

    for i in range(9):
        key = f'diagnosis_icd_{i}'
        if row.get(key) is not None:
            sentence += f" Diagnosed with {key}: {row[key]}."

    return sentence


with open("./train_data.json") as f:
    data_list = json.load(f)

texts = [create_sentence(row) for row in data_list]

labels_path = './data/y_train.csv'
y_train = pd.read_csv(labels_path, header=None)
y_train.fillna('None of the above', inplace=True)
labels = y_train[1].tolist()


def predict_heart_attack(text):
    return 'Heart attack' if 'Heart attack' in text else 'Not Heart attack'


labels = [predict_heart_attack(label) for label in labels]
label_to_int = {label: idx for idx, label in enumerate(set(labels))}
int_labels = [label_to_int[label] for label in labels]
y = np.array(int_labels, dtype=np.int32)

max_words = 10000
max_len = 100
embedding_dim = 128

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=max_len)

model = Sequential([
    Input(shape=(max_len,)),
    Embedding(input_dim=max_words, output_dim=embedding_dim),
    Bidirectional(LSTM(units=128, return_sequences=True)),
    Bidirectional(LSTM(units=128, return_sequences=True)),
    Bidirectional(LSTM(units=128, return_sequences=True)),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid'),
])

model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['AUC'])
model.summary()

with open("./test_data.json") as f:
    data_list = json.load(f)

texts = [create_sentence(row) for row in data_list]

labels_path = './data/y_test.csv'
y_test_df = pd.read_csv(labels_path, header=None)
y_test_df.fillna('None of the above', inplace=True)
labels = y_test_df[1].tolist()

labels = [predict_heart_attack(label) for label in labels]
label_to_int = {label: idx for idx, label in enumerate(set(labels))}
int_labels = [label_to_int[label] for label in labels]
y_test = np.array(int_labels, dtype=np.int32)

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X_test = pad_sequences(sequences, maxlen=max_len)

y_pred_prob = model.predict(X_test)

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('./roc.png')
