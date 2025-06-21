from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.quantization
import torch.nn.utils.prune as prune
import requests

from scipy.special import softmax

from datasets import load_dataset, load_metric
import pandas as pd
from sklearn.model_selection import train_test_split

tokenizer = BertTokenizer.from_pretrained('nreimers/MiniLM-L6-H384-uncased')
model = BertForSequenceClassification.from_pretrained('nreimers/MiniLM-L6-H384-uncased')
df2 = pd.read_csv('Classifier_dataset.csv', usecols = ["text_", "label_num"], dtype={"label_num" : "int8"})


train_df, test_df = train_test_split(df2, test_size=0.2, random_state=42)

#llm training
train_df.to_csv('train_dataset_llm.csv', index=False)
test_df.to_csv('test_dataset_llm.csv', index=False)



dataset = load_dataset('csv', data_files={'train': 'train_dataset_llm.csv', 'test': 'test_dataset_llm.csv'})

def tokenize_function(examples):
    return tokenizer(examples['text_'], padding="max_length", truncation=True)
def get_authenticity_score(text):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    scores = torch.softmax(outputs.logits, dim=1).numpy()
    authenticity_score = scores[0][1]
    print(scores)
    return authenticity_score

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.rename_column('label_num', 'labels')
# Set format for PyTorch
tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
training_args = TrainingArguments(
    output_dir="./results_llm_finetuning",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    
)
trainer.train()

model.save_pretrained("./fine-tuned-bert")
tokenizer.save_pretrained("./fine-tuned-bert")