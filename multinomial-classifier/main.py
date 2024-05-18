from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from datasets import load_dataset
import torch
import tensorflow as tf
import numpy as np

tweet_dataset = load_dataset(path='tweet_eval', name='emotion')  # Load the dataset
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')  # Instantiate the DistilBERT tokenizer
model = TFAutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=4)  # Instantiate the DistilBERT model

# print(tweet_dataset)
# print(tweet_dataset['train'])

print(f"Sequence samples:\n {tweet_dataset['train']['text'][:2]}")
print(f"Label samples:\n {tweet_dataset['train']['label'][:2]}")

category_names = {0: "anger", 1: "joy", 2: "optimism", 3: "sadness"}  # dict with category labels for conversion


def find_max_length(dataset):
    return len(max(dataset, key=lambda x: len(x.split())).split())
