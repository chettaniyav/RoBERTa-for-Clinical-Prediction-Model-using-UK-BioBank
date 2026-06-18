import os
import json
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from huggingface_hub import login, create_repo, upload_folder
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    BertTokenizerFast,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

tokenizer = AutoTokenizer.from_pretrained("YOUR_HF_USERNAME/YOUR_BASE_MODEL")
model = AutoModelForMaskedLM.from_pretrained("YOUR_HF_USERNAME/YOUR_BASE_MODEL")

data_path = './data/MLM_data.json'
with open(data_path) as f:
    data = json.load(f)


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


texts = [create_sentence(row) for row in data]

encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors='pt')
inputs = encodings['input_ids']
attention_mask = encodings['attention_mask']


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, attention_mask):
        self.inputs = inputs
        self.attention_mask = attention_mask

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs[idx],
            'attention_mask': self.attention_mask[idx],
        }


dataset = TextDataset(inputs, attention_mask)

model.resize_token_embeddings(len(tokenizer))

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
)

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
    learning_rate=5e-5,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-8,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = torch.nn.DataParallel(model)
model.to(device)

trainer.train()

model_save_path = "./mlm"
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

training_logs = trainer.state.log_history
train_losses = [log['loss'] for log in training_logs if 'loss' in log]

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Training Loss Over Steps MLM')
plt.legend()
plt.savefig("training_loss.png")

login(token=os.getenv("HF_TOKEN"))

hf_repo_name = "YOUR_HF_USERNAME/YOUR_MLM_MODEL"

tokenizer = BertTokenizerFast.from_pretrained(model_save_path)
model = BertForMaskedLM.from_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
model.save_pretrained(model_save_path)

create_repo(hf_repo_name, exist_ok=True)
upload_folder(
    repo_id=hf_repo_name,
    folder_path=model_save_path,
    commit_message="Initial commit of the trained model",
)

print(f"Model pushed to https://huggingface.co/{hf_repo_name}")
