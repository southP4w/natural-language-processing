from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from datasets import load_dataset
import torch
import tensorflow as tf
import numpy as np

tweet_dataset = load_dataset(path='tweet_eval', name='emotion')  # Load the dataset
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')  # Instantiate the DistilBERT tokenizer
model = TFAutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=4)  # Instantiate the DistilBERT model

print(tweet_dataset)
