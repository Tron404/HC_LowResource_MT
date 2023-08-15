import time
import numpy as np
import pandas as pd
import pickle
import math
import torch
import os
import evaluate
import sys

from datasets import Dataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from transformers import TrainingArguments, Trainer
from transformers import  XLMRobertaForSequenceClassification, XLMRobertaTokenizer, DataCollatorWithPadding

model_path = "./models/xlm-roberta-base"
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

file_id_pair = {id: file for id, file in enumerate(os.listdir("translation/complete"))}
data_id = int(sys.argv[1])
file_name = file_id_pair[data_id]

print(f"Working on file {file_name}")

accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy_score = accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    precision_score = precision.compute(predictions=predictions, references=labels)["precision"]
    recall_score = recall.compute(predictions=predictions, references=labels)["recall"]
    f1_score = f1.compute(predictions=predictions, references=labels)["f1"]

    return {"accruacy": accuracy_score, "precision":precision_score, "recall":recall_score, "f1":f1_score}

def tok_data(data):
    return tokenizer(data["claim"], truncation=True, padding="max_length", max_length=512)

limit = 50000

id2label = {0: "False", 1: "True"}
label2id = {"False": 0, "True": 1}

tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)
model = XLMRobertaForSequenceClassification.from_pretrained(model_path, num_labels = 2, id2label = id2label, label2id = label2id).to(device)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

text = pd.read_csv(f"./translation/complete/{file_name}", sep="|")["claim"].to_numpy().tolist()[:limit]
labels = pd.read_csv("./data/labels.csv").replace([True, False], [1, 0]).to_numpy().ravel()[:limit]

print(text[:50])

x_train, x_test, y_train, y_test = train_test_split(text, labels, test_size=0.25, random_state=42)
x_test, x_valid, y_test, y_valid = train_test_split(x_test, y_test, test_size=0.5, random_state=42)

ds_train = Dataset.from_dict({"claim":x_train, "label":y_train})
ds_test = Dataset.from_dict({"claim":x_test, "label":y_test})
ds_valid = Dataset.from_dict({"claim":x_valid, "label":y_valid})

tok_train = ds_train.map(tok_data, batched=True)
tok_test = ds_test.map(tok_data, batched=True)
tok_val = ds_valid.map(tok_data, batched=True)

training_args = TrainingArguments(
    output_dir = "finetuned_XLMRoBERTa",
    learning_rate = 1e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="no",
    push_to_hub=False,
    do_train = True,
    optim="adamw_torch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tok_train,
    eval_dataset=tok_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("Started training")
train_result = trainer.train()

print("Started testing")
eval_result = trainer.evaluate(eval_dataset=tok_test)

results = pd.DataFrame({"epoch": list(range(len(trainer.state.log_history))), "results":trainer.state.log_history})
results.to_pickle(f"./results/results_{file_name.split('.')[0]}.pickle")


