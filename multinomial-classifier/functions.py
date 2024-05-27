import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset


def find_max_length(dataset):
    return len(max(dataset, key=lambda x: len(x.split())).split())


def filter_dataset(dataset, num_words):
    return dataset.filter(lambda x: len(x["text"].split()) <= num_words)


def lr_decay(current_epoch):
    """
    Learning rate decay function
    :param current_epoch: the current epoch
    :return: learning rate decay
    """
    if current_epoch < 10:  # if the current epoch number is less than 10,
        return 1.0  # return a learning rate of 1.0
    return np.exp(-0.1 * (current_epoch - 10))  # else return an exponentially-decayed learning rate


def convert_to_tensors(tokenized_dataset, split):
    """
    Convert tokenized data to tensors
    :param tokenized_dataset: the preprocessed, tokenized dataset
    :param split: the data split to be converted to a tensor
    :return: the specified data split, in tensor format
    """
    input_ids = torch.tensor(tokenized_dataset[split]['input_ids'])
    attention_mask = torch.tensor(tokenized_dataset[split]['attention_mask'])
    labels = torch.tensor(tokenized_dataset[split]['label'])
    return TensorDataset(input_ids, attention_mask, labels)


def plot_training_results(train_loss, val_loss, test_loss, train_accuracy, val_accuracy, test_accuracy):
    """
    Function to plot training results
    :param train_loss: the training loss
    :param val_loss: the validation loss
    :param test_loss: the test loss
    :param train_accuracy: the training accuracy
    :param val_accuracy: the validation accuracy
    :param test_accuracy: the test accuracy
    """
    plt.figure(figsize=(12, 5))  # initialize the graph figure dimensions (width, height)

    # Losses graph:
    plt.subplot(1, 2, 1)  # subplot for losses
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.plot(test_loss, label="Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Over Epochs")
    plt.legend()  # graph legend

    # Accuracies graph:
    plt.subplot(1, 2, 2)  # subplot for accuracies
    plt.plot(train_accuracy, label="Train Accuracy")
    plt.plot(val_accuracy, label="Validation Accuracy")
    plt.plot(test_accuracy, label="Test Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Over Epochs")
    plt.legend()  # graph legend

    plt.tight_layout()  # adjust layout
    plt.show()  # show the plot
