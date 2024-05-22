import tensorflow as tf
from tensorflow.data import Dataset
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from datasets import load_dataset
import numpy as np
import functions as f
import matplotlib.pyplot as plt

# Load the dataset and model
tweet_dataset = load_dataset(path='tweet_eval', name='emotion')
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = TFAutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=4)

category_names = {0: "anger", 1: "joy", 2: "optimism", 3: "sadness"}

# Calculate maximum lengths
train_max_length = f.find_max_length(tweet_dataset["train"]["text"])
val_max_length = f.find_max_length(tweet_dataset["validation"]["text"])
test_max_length = f.find_max_length(tweet_dataset["test"]["text"])

num_words = 36
filtered_dataset = f.filter_dataset(tweet_dataset, num_words)


def tokenize_dataset(examples):
    return tokenizer(examples["text"], padding="max_length",
                     truncation=True, max_length=36)


tokenized_dataset = filtered_dataset.map(tokenize_dataset)

# Preparing datasets
train_features = tokenized_dataset["train"].remove_columns(["text", "label"]).with_format("tensorflow")
val_features = tokenized_dataset["validation"].remove_columns(["text", "label"]).with_format("tensorflow")
test_features = tokenized_dataset["test"].remove_columns(["text", "label"]).with_format("tensorflow")

train_features = {x: train_features[x] for x in tokenizer.model_input_names}
val_features = {x: val_features[x] for x in tokenizer.model_input_names}
test_features = {x: test_features[x] for x in tokenizer.model_input_names}

train_labels = tf.keras.utils.to_categorical(tokenized_dataset["train"]["label"])
val_labels = tf.keras.utils.to_categorical(tokenized_dataset["validation"]["label"])
test_labels = tf.keras.utils.to_categorical(tokenized_dataset["test"]["label"])

train_dataset = Dataset.from_tensor_slices((train_features, train_labels)).shuffle(len(train_features), seed=2).batch(8)
val_dataset = Dataset.from_tensor_slices((val_features, val_labels)).shuffle(len(train_features), seed=2).batch(8)
test_dataset = Dataset.from_tensor_slices((test_features, test_labels)).shuffle(len(train_features), seed=2).batch(8)

model.layers[0].trainable = False
model.summary()


def lr_decay(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * np.exp(-0.1 * epoch)


lr_scheduler = tf.keras.callbacks.LearningRateScheduler(schedule=lr_decay, verbose=1)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=tf.keras.metrics.CategoricalAccuracy())

history = model.fit(train_dataset, validation_data=val_dataset,
                    epochs=15, callbacks=[lr_scheduler])

model.evaluate(test_dataset)

predictions = model.predict(test_features)


def logits_to_class_names(predictions):
    predictions = tf.nn.softmax(predictions.logits)
    predictions = tf.argmax(predictions, axis=1).numpy()
    predictions = [category_names[prediction] for prediction in predictions]
    return predictions


test_batch = next(iter(test_dataset))[0]
sample_predictions = logits_to_class_names(model(test_batch))

for i in range(len(test_batch["input_ids"])):
    print(f"Tweet: {tokenizer.decode(test_batch['input_ids'][i])}")
    print(f"Predicted class: {sample_predictions[i]}\n")


def plot_training_results(history):
    plt.figure(figsize=(12, 5))

    # loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.legend()

    # accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['categorical_accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_categorical_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.legend()

    # output plot
    plt.tight_layout()
    plt.show()


plot_training_results(history)
