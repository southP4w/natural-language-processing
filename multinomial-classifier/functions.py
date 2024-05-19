
def find_max_length(dataset):
    return len(max(dataset, key=lambda x: len(x.split())).split())


def filter_dataset(dataset, num_words):
    return dataset.filter(lambda x: len(x["text"].split()) <= num_words)
