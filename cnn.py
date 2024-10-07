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


# Replace this with your actual token
token = "your_huggingface_token"
HfFolder.save_token(token)

# Load the data
with open("./train_data.json") as f:
    data_list = json.load(f)

texts = [create_sentence(row) for row in data_list]

labels_path = '/mnt/bmh01-rds/Jenkins_HDS_dissertations/n68517cv/data/y_train.csv'
y_train = pd.read_csv(labels_path, header=None)
y_train.fillna('None of the above', inplace=True)
labels = y_train[1].tolist()

# Function to predict if 'Heart Attack' is in the list


def predict_heart_attack(text):
    return 'Heart attack' if 'Heart attack' in text else 'Not Heart attack'


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

    # Bidirectional LSTM layer
    Bidirectional(LSTM(units=128, return_sequences=True)),

    # # Attention mechanism
    # Attention(),

    # Bidirectional LSTM layer
    Bidirectional(LSTM(units=128, return_sequences=True)),

    # Attention mechanism
    # Attention(),
    # Bidirectional LSTM layer
    Bidirectional(LSTM(units=128, return_sequences=True)),

    # Attention mechanism
    # Attention(),


    # Fully connected layer with dropout
    Dense(256, activation='relu'),
    Dropout(0.5),

    # Another fully connected layer
    Dense(128, activation='relu'),
    Dropout(0.5),

    # Output layer (for binary classification)
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='binary_crossentropy', metrics=['AUC'])

# Model summary
model.summary()


with open("./test_data.json") as f:
    data_list = json.load(f)

texts = [create_sentence(row) for row in data_list]

labels_path = '/mnt/bmh01-rds/Jenkins_HDS_dissertations/n68517cv/data/y_test.csv'
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


y_pred_prob = model.predict(X_test)

# Calculate the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Calculate the AUC
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC curve (area = {roc_auc:0.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
# plt.show()
plt.savefig('./roc.png')
