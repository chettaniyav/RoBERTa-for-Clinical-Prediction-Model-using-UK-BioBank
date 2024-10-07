from sklearn.metrics import roc_auc_score, average_precision_score
from tensorflow.keras.optimizers import AdamW
from sklearn.utils import resample
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Bidirectional, Attention, LayerNormalization
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout, Input
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, Input, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
from datetime import datetime
import numpy as np
from huggingface_hub import HfApi, HfFolder


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


# # Replace this with your actual token
# token = "your_huggingface_token"
# HfFolder.save_token(token)

# Load the data
with open("/mnt/bmh01-rds/Jenkins_HDS_dissertations/n68517cv/main/SUBERT/model/Fine-Tune/Test/St/train_data.json") as f:
    data_list = json.load(f)

texts = [create_sentence(row) for row in data_list]

labels_path = '/mnt/bmh01-rds/Jenkins_HDS_dissertations/n68517cv/data/y_train_new.csv'
y_train = pd.read_csv(labels_path, header=None)
y_train.fillna('None of the above', inplace=True)
labels = y_train[1].tolist()

# Function to predict if 'Heart Attack' is in the list

# condition = "Angina"

conditions = ["Angina", "Stroke", "Heart attack"]


for condition in conditions:
    print(f"Evaluating model for: {condition}")

    def predict_heart_attack(text):
        return condition if condition in text else f'Not {condition}'

    # Apply the function to all labels
    labels = [predict_heart_attack(label) for label in labels]

    # Create a mapping from text labels to integers
    label_to_int = {label: idx for idx, label in enumerate(set(labels))}
    int_labels = [label_to_int[label] for label in labels]

    # Convert labels to a numpy array with dtype=int32
    y = np.array(int_labels, dtype=np.int32)

    # Hyperparameters
    max_words = 10000  # Maximum number of words to keep, based on word frequency
    max_len = 100  # Maximum length of each sequence

    # Tokenizing the text
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    # Padding sequences to ensure uniform input size
    X = pad_sequences(sequences, maxlen=max_len)

    # Build the model
    # Build the model
    # model = Sequential([
    #     Input(shape=(max_len,)),  # Explicitly define input shape
    #     Embedding(input_dim=max_words, output_dim=128),

    #     Conv1D(filters=128, kernel_size=5, activation='relu'),
    #     Conv1D(filters=128, kernel_size=5, activation='relu'),
    #     Conv1D(filters=128, kernel_size=5, activation='relu'),

    #     GlobalMaxPooling1D(),
    #     Dropout(0.5),

    #     Dense(128, activation='relu'),
    #     Dropout(0.5),

    #     Dense(1, activation='sigmoid')  # For binary classification
    # ])

    # # Compile the model
    # model.compile(optimizer='adam', loss='binary_crossentropy',
    #               metrics=['accuracy'])

    # model.summary()

    # # Train the model
    # model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

    embedding_dim = 128  # Dimension of the embedding vector

    # # Build the model
    # model = Sequential([
    #     # Input layer (shape of each input sequence)
    #     Input(shape=(max_len,)),

    #     # Embedding layer (outputs a 3D tensor with shape (batch_size, max_len, embedding_dim))
    #     Embedding(input_dim=max_words, output_dim=embedding_dim),

    #     # RNN layer (SimpleRNN is one of the basic types of RNN)
    #     # return_sequences=False to only return the output for the last timestep
    #     Bidirectional(SimpleRNN(units=128, return_sequences=False)),

    #     # Fully connected layer with dropout
    #     Dense(128, activation='relu'),
    #     Dropout(0.5),

    #     # Output layer (for binary classification)
    #     Dense(1, activation='sigmoid')
    # ])

    # # Compile the model
    # model.compile(optimizer='adam', loss='binary_crossentropy',
    #               metrics=['accuracy'])

    # # Model summary
    # model.summary()
    # model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

    # Build the model
    model = Sequential([
        # Input layer
        Input(shape=(max_len,)),

        # Embedding layer
        Embedding(input_dim=max_words, output_dim=embedding_dim),

        # Bidirectional LSTM layers with dropout in between
        Bidirectional(LSTM(units=128, return_sequences=True,
                           dropout=0.2, recurrent_dropout=0.2)),
        Dropout(0.2),  # Dropout between Bidirectional LSTM layers

        Bidirectional(LSTM(units=128, return_sequences=True,
                           dropout=0.2, recurrent_dropout=0.2)),
        Dropout(0.2),  # Dropout between Bidirectional LSTM layers

        Bidirectional(LSTM(units=128, return_sequences=True,
                           dropout=0.2, recurrent_dropout=0.2)),
        Dropout(0.2),  # Dropout between Bidirectional LSTM layers

        # Set return_sequences=False for the final LSTM layer
        Bidirectional(LSTM(units=128, return_sequences=False,
                           dropout=0.2, recurrent_dropout=0.2)),

        # Fully connected layer with dropout
        Dense(256, activation='relu'),
        Dropout(0.5),

        # Another fully connected layer with dropout
        Dense(128, activation='relu'),
        Dropout(0.5),

        # Output layer (for binary classification)
        Dense(1, activation='sigmoid')
    ])
    # Compile the model
    model.compile(optimizer=AdamW(learning_rate=1e-4, weight_decay=1e-5),
                  loss='binary_crossentropy', metrics=['AUC'])

    # Model summary
    model.summary()

    model.fit(X, y, epochs=10, batch_size=1024, validation_split=0.2)

    with open("/mnt/bmh01-rds/Jenkins_HDS_dissertations/n68517cv/main/SUBERT/model/Fine-Tune/Test/St/test_data.json") as f:
        data_list = json.load(f)

    texts = [create_sentence(row) for row in data_list]

    labels_path = '/mnt/bmh01-rds/Jenkins_HDS_dissertations/n68517cv/data/y_test_new.csv'
    y_train = pd.read_csv(labels_path, header=None)
    y_train.fillna('None of the above', inplace=True)
    labels = y_train[1].tolist()

    # Apply the function to all labels
    labels = [predict_heart_attack(label) for label in labels]

    # Create a mapping from text labels to integers
    label_to_int = {label: idx for idx, label in enumerate(set(labels))}
    int_labels = [label_to_int[label] for label in labels]

    # Convert labels to a numpy array with dtype=int32
    y_test = np.array(int_labels, dtype=np.int32)

    # Hyperparameters
    max_words = 10000  # Maximum number of words to keep, based on word frequency
    max_len = 100  # Maximum length of each sequence

    # Tokenizing the text
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    # Padding sequences to ensure uniform input size
    X_test = pad_sequences(sequences, maxlen=max_len)

    y_pred_prob = model.predict(X_test).ravel()

    # # Calculate the ROC curve
    # fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

    # # Calculate the AUC
    # roc_auc = auc(fpr, tpr)

    # # Plot the ROC curve
    # plt.figure()
    # plt.plot(fpr, tpr, color='darkorange', lw=2,
    #          label=f'ROC curve (area = {roc_auc:0.2f})')
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic')
    # plt.legend(loc="lower right")
    # # plt.show()
    # plt.savefig('./roc.png')
    # Define a function to compute AUROC and AUPRC with bootstrapping

    def compute_metrics_with_ci(y_true, y_pred_prob, num_bootstrap=1000, ci_percentile=95):
        auroc_scores = []
        auprc_scores = []

        n_samples = len(y_true)

        for i in range(num_bootstrap):
            # Resample the dataset with replacement
            indices = resample(np.arange(n_samples),
                               replace=True, n_samples=n_samples)
            y_true_resampled = y_true[indices]
            y_pred_prob_resampled = y_pred_prob[indices]

            # Calculate AUROC
            fpr, tpr, _ = roc_curve(y_true_resampled, y_pred_prob_resampled)
            auroc = auc(fpr, tpr)
            auroc_scores.append(auroc)

            # Calculate AUPRC
            precision, recall, _ = precision_recall_curve(
                y_true_resampled, y_pred_prob_resampled)
            auprc = average_precision_score(
                y_true_resampled, y_pred_prob_resampled)
            auprc_scores.append(auprc)

        # Calculate the mean and confidence intervals
        auroc_mean = np.mean(auroc_scores)
        auroc_ci_lower = np.percentile(auroc_scores, (100 - ci_percentile) / 2)
        auroc_ci_upper = np.percentile(
            auroc_scores, 100 - (100 - ci_percentile) / 2)

        auprc_mean = np.mean(auprc_scores)
        auprc_ci_lower = np.percentile(auprc_scores, (100 - ci_percentile) / 2)
        auprc_ci_upper = np.percentile(
            auprc_scores, 100 - (100 - ci_percentile) / 2)

        return {
            'auroc_mean': auroc_mean,
            'auroc_ci': (auroc_ci_lower, auroc_ci_upper),
            'auprc_mean': auprc_mean,
            'auprc_ci': (auprc_ci_lower, auprc_ci_upper)
        }

    def bootstrap_metrics(predictions, true_labels, n_bootstraps=10000, ci=95):
        """
        Calculate AUPRC and AUROC with bootstrapping and return the confidence intervals.
        """
        auprc_scores = []
        auroc_scores = []
        rng = np.random.RandomState(42)  # For reproducibility
        n_bootstraps = len(true_labels)
        # Use min between len(labels) and n_bootstraps if len(labels) is smaller

        for i in range(n_bootstraps):
            indices = rng.randint(0, len(predictions), len(predictions))

            # Ensure at least two classes are present in the bootstrapped sample
            if len(np.unique(true_labels[indices])) < 2:
                continue  # Skip this bootstrap sample

            try:
                # Calculate AUPRC and AUROC for this bootstrap sample
                auprc_score = average_precision_score(
                    true_labels[indices], predictions[indices])
                auprc_scores.append(auprc_score)

                auroc_score = roc_auc_score(
                    true_labels[indices], predictions[indices])
                auroc_scores.append(auroc_score)
            except Exception as e:
                print(f"Error in bootstrap sample {i}: {e}")
                continue

        # Ensure that valid scores were collected
        if len(auprc_scores) == 0 or len(auroc_scores) == 0:
            print("No valid AUPRC or AUROC scores were calculated during bootstrapping.")
            return (None, None, None), (None, None, None)

        # Calculate means and confidence intervals
        mean_auprc = np.mean(auprc_scores)
        lower_auprc = np.percentile(auprc_scores, (100 - ci) / 2)
        upper_auprc = np.percentile(auprc_scores, 100 - (100 - ci) / 2)

        mean_auroc = np.mean(auroc_scores)
        lower_auroc = np.percentile(auroc_scores, (100 - ci) / 2)
        upper_auroc = np.percentile(auroc_scores, 100 - (100 - ci) / 2)

        return (mean_auprc, lower_auprc, upper_auprc), (mean_auroc, lower_auroc, upper_auroc)

    # Assuming your model and data preprocessing code is already here...

    # Make predictions
    y_pred_prob = model.predict(X_test).ravel()
    # Compute AUROC and AUPRC with confidence intervals
    # metrics = compute_metrics_with_ci(y_test, y_pred_prob)

    # # Print out the AUROC and AUPRC with confidence intervals
    # print(
    #     f'AUC (ROC): {metrics["auroc_mean"]:.4f} (95% CI: {metrics["auroc_ci"][0]:.4f} - {metrics["auroc_ci"][1]:.4f})')
    # print(
    #     f'AUPRC: {metrics["auprc_mean"]:.4f} (95% CI: {metrics["auprc_ci"][0]:.4f} - {metrics["auprc_ci"][1]:.4f})')

    (auprc_mean, auprc_lower, auprc_upper), (auroc_mean, auroc_lower,
                                             auroc_upper) = bootstrap_metrics(y_pred_prob, y_test
                                                                              )

    print(
        f"AUPRC: {auprc_mean:.4f} ± {auprc_mean - auprc_lower:.4f} (95% CI: [{auprc_lower:.4f}, {auprc_upper:.4f}])")
    print(
        f"AUROC: {auroc_mean:.4f} ± {auroc_mean - auroc_lower:.4f} (95% CI: [{auroc_lower:.4f}, {auroc_upper:.4f}])")

    # # Plot the final ROC and Precision-Recall curves with mean AUC and AUPRC
    # fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    # plt.figure()
    # plt.plot(fpr, tpr, color='darkorange', lw=2,
    #          label=f'ROC curve (mean AUC = {metrics["auroc_mean"]:0.2f})')
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic')
    # plt.legend(loc="lower right")
    # plt.savefig('./roc_with_ci.png')

    # precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
    # plt.figure()
    # plt.plot(recall, precision, color='blue', lw=2,
    #          label=f'Precision-Recall curve (mean AUPRC = {metrics["auprc_mean"]:0.2f})')
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title('Precision-Recall Curve')
    # plt.legend(loc="lower left")
    # plt.savefig('./prc_with_ci.png')
