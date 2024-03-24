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
        self.V = {}  # unigrams
        self.W = []  # full set of tokens

    def train(self, training_corpus: iter):
        with open(file=training_corpus, mode='r', encoding='utf-8') as T:
            for s in T:
                W_ = []
                for w in s.strip().split():
                    if w != '<s>':
                        W_.append(w)
                self.W.extend(W_)
                for w in W_:
                    self.V[w] = self.V.get(w, 0) + 1

    def compute_probability_raw(self, word: str) -> float:
        return self.V.get(word, 0) / len(self.W)

    def compute_probability_log(self, sentence: iter) -> float:
        s = ["<s>"] + sentence.strip().lower().split() + ["</s>"]
        log_2_p = 0
        for w in s:
            if w not in self.V:
                w = "<unk>"
            p = self.compute_probability_raw(w)
            try:
                print(f"Unigram: ({w})\n\tRAW Probability: {p * 100:.4f}%\n\tlog_2 Probability: {math.log2(p):.4f}%")
                log_2_p += math.log2(p)
            except ValueError:
                print(f"UNIGRAM: ({w})\n\tRAW PROBABILITY: {p} ZERO")

        return log_2_p

    def compute_perplexity_of_sentence(self, sentence: iter) -> float:
        s = ["<s>"] + sentence.strip().lower().split() + ["</s>"]
        log_2_p = 0
        for w in s:
            if w not in self.V:
                w = "<unk>"
            p = self.compute_probability_raw(w)
            log_2_p += math.log2(p)
        PPL = 2**(-log_2_p / len(s))

        return PPL

    def compute_perplexity_of_corpus(self, corpus: iter) -> float:
        log_2_p = 0
        total_words = 0
        with open(file=corpus, mode='r', encoding='utf-8') as T:
            for c in T:
                s = ["<s>"] + c.strip().lower().split() + ["</s>"]
                total_words += len(s)
                for w in s:
                    if w not in self.V:
                        w = "<unk>"
                    p = self.compute_probability_raw(w)
                    log_2_p += math.log2(p)
        PPL = 2**(-log_2_p / total_words)

        return PPL

    def get_unigram_count(self) -> int:
        return len(self.V)

    def get_token_count(self) -> int:
        return len(self.W)
