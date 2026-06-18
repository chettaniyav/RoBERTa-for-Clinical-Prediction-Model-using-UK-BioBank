import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, auc, precision_recall_curve
from sklearn.utils import resample
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, SimpleRNN, Dense, Dropout, Bidirectional


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


with open("./data/train_data.json") as f:
    data_list = json.load(f)

texts = [create_sentence(row) for row in data_list]

labels_path = './data/y_train.csv'
y_train = pd.read_csv(labels_path, header=None)
y_train.fillna('None of the above', inplace=True)
labels = y_train[1].tolist()

conditions = ["Angina", "Stroke", "Heart attack"]

for condition in conditions:
    print(f"Evaluating model for: {condition}")

    def predict_condition(text):
        return condition if condition in text else f'Not {condition}'

    labels = [predict_condition(label) for label in labels]
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
        Bidirectional(SimpleRNN(units=128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)),
        Dropout(0.2),
        Bidirectional(SimpleRNN(units=128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)),
        Dropout(0.2),
        Bidirectional(SimpleRNN(units=128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)),
        Dropout(0.2),
        Bidirectional(SimpleRNN(units=128, return_sequences=False, dropout=0.2, recurrent_dropout=0.2)),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid'),
    ])

    model.compile(optimizer=AdamW(learning_rate=1e-4, weight_decay=1e-5),
                  loss='binary_crossentropy', metrics=['AUC'])
    model.summary()
    model.fit(X, y, epochs=10, batch_size=1024, validation_split=0.2)

    with open("./data/test_data.json") as f:
        data_list = json.load(f)

    texts = [create_sentence(row) for row in data_list]

    labels_path = './data/y_test.csv'
    y_test_df = pd.read_csv(labels_path, header=None)
    y_test_df.fillna('None of the above', inplace=True)
    labels = y_test_df[1].tolist()
    labels = [predict_condition(label) for label in labels]
    label_to_int = {label: idx for idx, label in enumerate(set(labels))}
    int_labels = [label_to_int[label] for label in labels]
    y_test = np.array(int_labels, dtype=np.int32)

    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    X_test = pad_sequences(sequences, maxlen=max_len)

    y_pred_prob = model.predict(X_test).ravel()

    def bootstrap_metrics(predictions, true_labels, n_bootstraps=10000, ci=95):
        auprc_scores = []
        auroc_scores = []
        rng = np.random.RandomState(42)
        n_bootstraps = len(true_labels)

        for i in range(n_bootstraps):
            indices = rng.randint(0, len(predictions), len(predictions))
            if len(np.unique(true_labels[indices])) < 2:
                continue
            try:
                auprc_scores.append(average_precision_score(true_labels[indices], predictions[indices]))
                auroc_scores.append(roc_auc_score(true_labels[indices], predictions[indices]))
            except Exception as e:
                print(f"Error in bootstrap sample {i}: {e}")
                continue

        if len(auprc_scores) == 0 or len(auroc_scores) == 0:
            print("No valid AUPRC or AUROC scores were calculated during bootstrapping.")
            return (None, None, None), (None, None, None)

        mean_auprc = np.mean(auprc_scores)
        lower_auprc = np.percentile(auprc_scores, (100 - ci) / 2)
        upper_auprc = np.percentile(auprc_scores, 100 - (100 - ci) / 2)

        mean_auroc = np.mean(auroc_scores)
        lower_auroc = np.percentile(auroc_scores, (100 - ci) / 2)
        upper_auroc = np.percentile(auroc_scores, 100 - (100 - ci) / 2)

        return (mean_auprc, lower_auprc, upper_auprc), (mean_auroc, lower_auroc, upper_auroc)

    y_pred_prob = model.predict(X_test).ravel()
    (auprc_mean, auprc_lower, auprc_upper), (auroc_mean, auroc_lower, auroc_upper) = bootstrap_metrics(
        y_pred_prob, y_test)

    print(f"AUPRC: {auprc_mean:.4f} ± {auprc_mean - auprc_lower:.4f} (95% CI: [{auprc_lower:.4f}, {auprc_upper:.4f}])")
    print(f"AUROC: {auroc_mean:.4f} ± {auroc_mean - auroc_lower:.4f} (95% CI: [{auroc_lower:.4f}, {auroc_upper:.4f}])")
