import math


class BigramModel:
    def __init__(self):
        self.V_0 = {}  # unigrams
        self.V_1 = {}  # bigrams
        self.W = []  # full set of bigram tokens

    def train(self, training_corpus: iter):
        with open(file=training_corpus, mode='r', encoding='utf-8') as T:
            for s in T:
                W_ = []
                for w in s.strip().split():
                    if w != '<s>':
                        W_.append(w)
                for i in range(len(W_) - 1):
                    bigram = (W_[i], W_[i + 1])
                    self.W.append(bigram)  # Append bigram tokens to W
                    self.V_0[W_[i]] = self.V_0.get(W_[i], 0) + 1
                    self.V_1[bigram] = self.V_1.get(bigram, 0) + 1

    def compute_probability_raw(self, word_0: str, word_1: str) -> float:
        return (self.V_1.get((word_0, word_1), 0)) / (self.V_0.get(word_0, 1))

    def compute_probability_log(self, sentence: iter) -> float:
        s = ["<s>"] + sentence.strip().lower().split() + ["</s>"]
        log_2_p = 0
        for i in range(len(s) - 1):
            w_0, w_1 = s[i], s[i + 1]
            if w_0 not in self.V_0:
                w_0 = "<unk>"
            if (w_0, w_1) not in self.V_1:
                w_1 = "<unk>"
            p = self.compute_probability_raw(w_0, w_1)
            try:
                log_2_p += math.log2(p)
                print(f"Bigram: ({w_1}| {w_0}),\n\tRaw Probability: {p * 100:.4f}%\n\tlog_2 Probability: {math.log2(p):.4f}%")
            except ValueError:
                print(f"BIGRAM: ({w_1}| {w_0})\n\tRAW PROBABILITY: {p} ZERO")

        return log_2_p

    def compute_probability_smoothed(self, sentence: iter) -> float:
        s = ["<s>"] + sentence.strip().lower().split() + ["</s>"]
        # s = ["<s>"] + sentence.strip().lower().split() + ["</s>"]
        log_2_p = 0
        for i in range(len(s) - 1):
            w_0, w_1 = s[i], s[i + 1]
            if w_0 not in self.V_0:
                w_0 = "<unk>"
            if (w_0, w_1) not in self.V_1:
                w_1 = "<unk>"
            p = self.smooth_probabilities(w_0, w_1)
            try:
                log_2_p += math.log2(p)
                print(f"Bigram: ({w_1}| {w_0})\n\tSmoothed Probability: {p * 100:.4f}%\n\tSmoothed log_2 Probability: {math.log2(p):.4f}%")
            except ValueError:
                print(f"BIGRAM: ({w_1}| {w_0})\n\tSMOOTHED PROBABILITY: {p}% ZERO")

        return log_2_p

    def compute_perplexity_of_sentence(self, sentence: iter) -> float | str:
        s = ["<s>"] + sentence.strip().lower().split() + ["</s>"]
        log_2_p = 0
        for i in range(len(s) - 1):
            w_0, w_1 = s[i], s[i + 1]
            if w_0 not in self.V_0:
                w_0 = "<unk>"
            p = self.compute_probability_raw(w_0, w_1)
            try:
                log_2_p += math.log2(p)
            except ValueError:
                return "UNDEFINED! There exists a bigram tuple with a 0 value (cannot divide by zero)!"
        PPL = 2**(-log_2_p / len(s) - 1)

        return PPL

    def compute_perplexity_smoothed_of_sentence(self, sentence: iter) -> float:
        s = ["<s>"] + sentence.strip().lower().split() + ["</s>"]
        log_2_p = 0
        for i in range(len(s) - 1):
            w_0, w_1 = s[i], s[i + 1]
            if w_0 not in self.V_0:
                w_0 = "<unk>"
            p = self.smooth_probabilities(w_0, w_1)
            log_2_p += math.log2(p)
        PPL = 2**(-log_2_p / len(s) - 1)

        return PPL

    def compute_perplexity_of_corpus(self, corpus: iter) -> float | str:
        log_2_p = 0
        total_bigrams = 0
        with open(file=corpus, mode='r', encoding='utf-8') as T:
            for c in T:
                s = ["<s>"] + c.strip().lower().split() + ["</s>"]
                total_bigrams += len(s) - 1
                for i in range(len(s) - 1):
                    w_0, w_1 = s[i], s[i + 1]
                    if w_0 not in self.V_0:
                        w_0 = "<unk>"
                    if (w_0, w_1) not in self.V_1:
                        w_1 = "<unk>"
                    p = self.compute_probability_raw(w_0, w_1)
                    try:
                        log_2_p += math.log2(p)
                    except ValueError:
                        return "UNDEFINED! There exists a bigram tuple with a 0 value (cannot divide by zero)!"
        PPL = 2**(-log_2_p / total_bigrams)

        return PPL

    def compute_perplexity_smoothed_of_corpus(self, corpus: iter) -> float | str:
        log_2_p = 0
        total_bigrams = 0
        with open(file=corpus, mode='r', encoding='utf-8') as T:
            for c in T:
                s = ["<s>"] + c.strip().lower().split() + ["</s>"]
                total_bigrams += len(s) - 1
                for i in range(len(s) - 1):
                    w_0, w_1 = s[i], s[i + 1]
                    if w_0 not in self.V_0:
                        w_0 = "<unk>"
                    if (w_0, w_1) not in self.V_1:
                        w_1 = "<unk>"
                    p = self.smooth_probabilities(w_0, w_1)
                    try:
                        log_2_p += math.log2(p)
                    except ValueError:
                        return "UNDEFINED! There exists a bigram tuple with a 0 value (cannot divide by zero)!"
        PPL = 2**(-log_2_p / total_bigrams)

        return PPL

    def smooth_probabilities(self, word_0, word_1) -> float:
        return (self.V_1.get((word_0, word_1), 0) + 1) / (self.V_0.get(word_0, 0) + len(self.V_0))

    def get_unigram_count(self) -> int:
        return len(self.V_0)

    def get_bigram_count(self) -> int:
        return len(self.V_1)

    def get_token_count(self) -> int:
        return len(self.W)
