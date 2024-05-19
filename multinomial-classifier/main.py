""" Source: https://kedion.medium.com/fine-tuning-nlp-models-with-hugging-face-f92d55949b66 """

from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from datasets import load_dataset
import tensorflow.python.keras.utils.np_utils as tf
import numpy as np
import functions as f

tweet_dataset = load_dataset(path='tweet_eval', name='emotion')  # Load the dataset
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')  # Instantiate the DistilBERT tokenizer
model = TFAutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=4)  # Instantiate the DistilBERT model

# print(tweet_dataset)
# print(tweet_dataset['train'])

# print(f"Sequence samples:\n {tweet_dataset['train']['text'][:2]}")
# print(f"Label samples:\n {tweet_dataset['train']['label'][:2]}")

category_names = {0: "anger", 1: "joy", 2: "optimism", 3: "sadness"}  # dict with category labels for conversion

train_max_length = f.find_max_length(tweet_dataset["train"]["text"])  # get the longest sequence in the training set
val_max_length = f.find_max_length(tweet_dataset["validation"]["text"])  # get the longest sequence in the validation set
test_max_length = f.find_max_length(tweet_dataset["test"]["text"])  # get the longest sequence in the test set

# print(f"Longest sequence in train set has {train_max_length} words")  # print longest sequence in training set
# print(f"Longest sequence in val set has {val_max_length} words")  # print longest sequence in validation set
# print(f"Longest sequence in test set has {test_max_length} words")  # print longest sequence in test set

num_words = 36

filtered_dataset = f.filter_dataset(tweet_dataset, num_words)


def tokenize_dataset(examples):
    return tokenizer(examples["text"], padding="max_length",
                     truncation=True, max_length=36)


tokenized_dataset = filtered_dataset.map(tokenize_dataset)

# print(tokenized_dataset)
# print(tokenized_dataset["train"][0])

# Removing "text" and "label" columns from our data splits to craft features for the model:
train_features = tokenized_dataset["train"].remove_columns(["text", "label"]).with_format("tensorflow")
val_features = tokenized_dataset["validation"].remove_columns(["text", "label"]).with_format("tensorflow")
test_features = tokenized_dataset["test"].remove_columns(["text", "label"]).with_format("tensorflow")

# Converting our features to TF Tensors
train_features = {x: train_features[x] for x in tokenizer.model_input_names}
val_features = {x: val_features[x] for x in tokenizer.model_input_names}
test_features = {x: test_features[x] for x in tokenizer.model_input_names}

# print(tokenizer.model_input_names)
# print(train_features)

train_labels = tf.to_categorical(tokenized_dataset["train"]["label"])
val_labels = tf.to_categorical(tokenized_dataset["validation"]["label"])
test_labels = tf.to_categorical(tokenized_dataset["test"]["label"])

print(train_labels[:5])
