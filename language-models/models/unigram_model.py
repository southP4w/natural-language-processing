import math


class UnigramModel:
    """
    T   =   the corpus (text file)
    V   =   vocabulary: all word types in the corpus (no repeats)
    W   =   tokens: all word instances in the corpus (has repeats)
    W_  =   temporary list for storing the words in the current sentence
    s   =   sentence: a single sentence
    w   =   word:   a single word token
    """
    def __init__(self):
        self.V = {}
        self.W = []

    def train(self, training_corpus):
        with open(file=training_corpus, mode='r', encoding='utf-8') as T:
            for s in T:
                W_ = []
                for w in s.strip().split():
                    if w != '<s>':
                        W_.append(w)
                self.W.extend(W_)
                for w in W_:
                    self.V[w] = self.V.get(w, 0) + 1

    def unigram_count(self):
        return len(self.V)

    def token_count(self):
        return len(self.W)
