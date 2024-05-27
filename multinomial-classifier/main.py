import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import functions as f
from torch.optim.lr_scheduler import LambdaLR  # learning rate scheduler


def tokenize_dataset(dataset):
    """
    Tokenize a dataset and return a tokenized version of the dataset.
    :param dataset: Dataset to be tokenized.
    :return: tokenized version of the passed dataset
    """
    return tokenizer(dataset['text'], padding='max_length', truncation=True, max_length=num_words)


def logits_to_class_names(logits):
    """
    Convert logits to class names
    :param logits: the logits to be converted
    :return: the converted class names
    """
    predictions = torch.nn.functional.softmax(logits, dim=-1)  # apply softmax to logits
    predictions = torch.argmax(predictions, dim=1).cpu().numpy()  # get the index of the largest value
    return [category_names[pred] for pred in predictions]  # class names corresponding to the predictions


def evaluate(model, dataloader, device):
    """
    Evaluate the model on a given dataloader
    :param model: The model to be evaluated.
    :param dataloader: The dataloader for evaluation.
    :param device: The device to run the evaluation on.
    :return: The average loss and accuracy of the evaluation.
    """
    model.eval()  # set the model to evaluation mode
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():  # disable gradient calculation
        for batch in dataloader:
            b_input_ids, b_input_mask, b_labels = [x.to(device) for x in batch]  # move the batch to the device
            outputs = model(input_ids=b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs.loss
            total_loss += loss.item()  # accumulate the total loss
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            correct_predictions += (predictions == b_labels).sum().item()  # accumulate the correct predictions
            total_predictions += b_labels.size(0)  # accumulate the total predictions
    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_predictions
    return avg_loss, accuracy


tweet_dataset = load_dataset(path='tweet_eval', name='emotion')
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=4)

category_names = {0: 'anger', 1: 'joy', 2: 'optimism', 3: 'sadness'}

# Find the max length of each text:
train_max_length = f.find_max_length(tweet_dataset['train']['text'])
val_max_length = f.find_max_length(tweet_dataset['validation']['text'])
test_max_length = f.find_max_length(tweet_dataset['test']['text'])

num_words = max(train_max_length, val_max_length, test_max_length)  # dynamically get length of longest word sequence
# num_words = 36  # length of longest word sequence (this was originally pre-determined)
filtered_dataset = f.filter_dataset(tweet_dataset, num_words)
tokenized_dataset = filtered_dataset.map(tokenize_dataset, batched=True)

# convert each data split to a tensor:
train_dataset = f.convert_to_tensors(tokenized_dataset, 'train')
val_dataset = f.convert_to_tensors(tokenized_dataset, 'validation')
test_dataset = f.convert_to_tensors(tokenized_dataset, 'test')

batch_size = 16  # larger batch sizes utilize more RAM/VRAM

# Define the DataLoaders:
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

for param in model.base_model.parameters():  # freeze the base model parameters
    param.requires_grad = False

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # use the 'Adam' optimizer
scheduler = LambdaLR(optimizer, lr_lambda=f.lr_decay)  # define the learning rate scheduler
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)  # pass the model to the selected device

# Initialize loss and accuracy histories:
train_loss_history = []
val_loss_history = []
test_loss_history = []
train_accuracy_history = []
val_accuracy_history = []
test_accuracy_history = []

epochs = 150
for epoch in range(epochs):
    model.train()  # set the model to training mode
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    for batch in train_dataloader:
        b_input_ids, b_input_mask, b_labels = [x.to(device) for x in batch]
        model.zero_grad()  # zero-out the model gradients
        outputs = model(input_ids=b_input_ids, attention_mask=b_input_mask, labels=b_labels)  # get the model outputs

        # Accumulate and backpropagate the loss:
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()

        optimizer.step()  # step the optimizer
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        correct_predictions += (predictions == b_labels).sum().item()  # accumulate correct predictions
        total_predictions += b_labels.size(0)  # accumulate total predictions
    scheduler.step()  # step the scheduler

    # Calculate the average training and accuracy losses and append them to their respective histories:
    avg_train_loss = total_loss / len(train_dataloader)
    train_loss_history.append(avg_train_loss)
    train_accuracy = correct_predictions / total_predictions
    train_accuracy_history.append(train_accuracy)

    # Evaluate on the test and validation sets:
    avg_val_loss, val_accuracy = evaluate(model, val_dataloader, device)
    val_loss_history.append(avg_val_loss)
    val_accuracy_history.append(val_accuracy)
    avg_test_loss, test_accuracy = evaluate(model, test_dataloader, device)
    test_loss_history.append(avg_test_loss)
    test_accuracy_history.append(test_accuracy)

    print(f"\nEpoch {epoch + 1}/{epochs}")
    print(f"Train loss: {avg_train_loss:.4f}")
    print(f"Validation loss: {avg_val_loss:.4f}")
    print(f"Test loss: {avg_test_loss:.4f}")
    print("==========================================")
    print(f"Train accuracy: {train_accuracy:.4f}")
    print(f"Validation accuracy: {val_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

test_batch = next(iter(test_dataloader))  # get a batch from the test dataloader
test_input_ids, test_input_mask, _ = [x.to(device) for x in test_batch]
sample_logits = model(test_input_ids, attention_mask=test_input_mask).logits  # the sample logits
sample_predictions = logits_to_class_names(sample_logits)  # the sample predictions

for i in range(len(test_input_ids)):
    print(f"Tweet: {tokenizer.decode(test_input_ids[i], skip_special_tokens=True)}")
    print(f"Predicted class: {sample_predictions[i]}\n")

f.plot_training_results(train_loss_history, val_loss_history, test_loss_history, train_accuracy_history,
                        val_accuracy_history, test_accuracy_history)
