from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset

# Use a Llama checkpoint (make sure you have access to it)
llama_checkpoint = "meta-llama/Llama-2-7b-hf"

tokenizer = AutoTokenizer.from_pretrained(llama_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(llama_checkpoint, num_labels=2)

df2 = pd.read_csv('Classifier_dataset.csv', usecols=["text_", "label_num"], dtype={"label_num": "int8"})

train_df, test_df = train_test_split(df2, test_size=0.2, random_state=42)

train_df.to_csv('train_dataset_llm.csv', index=False)
test_df.to_csv('test_dataset_llm.csv', index=False)

dataset = load_dataset('csv', data_files={'train': 'train_dataset_llm.csv', 'test': 'test_dataset_llm.csv'})

def tokenize_function(examples):
    return tokenizer(examples['text_'], padding="max_length", truncation=True, max_length=512)

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
tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

training_args = TrainingArguments(
    output_dir="./results_llm_finetuning",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,  # Llama models are large, use small batch size unless you have lots of VRAM
    per_device_eval_batch_size=2,
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

model.save_pretrained("./fine-tuned-llama")
tokenizer.save_pretrained("./fine-tuned-llama")