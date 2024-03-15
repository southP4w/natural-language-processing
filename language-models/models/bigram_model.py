import math


class BigramModel:
    """

    """
    def __init__(self):
        self.bigram_counts = {}
        self.unigram_counts = {}

    def train(self, file_name):
        with open(file=file_name, mode='r', encoding='utf-8') as file:
            for line in file:
                words = [word for word in line.strip().split() if word != '<s>']
                for i in range(len(words) - 1):
                    bigram = (words[i], words[i + 1])
                    self.bigram_counts[bigram] = self.bigram_counts.get(bigram, 0) + 1
                    self.unigram_counts[words[i]] = self.unigram_counts.get(words[i], 0) + 1
