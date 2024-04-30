import utilities as util
import preprocess as prep
import os
import math


def documents_per_class(c, D):
    n = 0
    i = 0
    for _ in D:
        if D[i][1] == c:  # D[d[vectorized_document][CATEGORY]] --> if category for this document == c:
            n += 1
        i += 1

    return n


def define_vocabulary(D):
    V = {}
    for d in D:
        for w in d[0]:
            V[w] = V.get(w, 0) + 1

    return V


def initialize_bigdocs_list(C):
    B = []
    for c in C:
        B.append(c)

    return B


def zero_out_prior_probabilities(C):
    zero_probs = [0 for _ in C]

    return zero_probs


def populate_bigdoc(c, D):
    i = 0
    b = []
    for d in D:
        if D[i][1] == c:
            for w in d[0]:
                b.append(w)
        i += 1

    return b


class NBClassifier:

    def __init__(self):
        self.N_doc = 0  # number of documents in D
        self.N_c = 0  # number of documents from D in class c
        self.D = []  # set of documents to be trained on
        self.C = []  # list of categories for the documents
        self.V = {}  # dict to store the vocabulary generated for the documents
        self.logprior = []  # list of prior probabilities for the categories, P(c)
        self.loglikelihood = {}  # dict to store the probabilities of assigned category for each word, P(word | category)
        self.bigdoc = []  # one list for storing all words in all documents for each category

    def train_naive_bayes(self, D, C):
        self.logprior = zero_out_prior_probabilities(C)
        self.bigdoc = initialize_bigdocs_list(C)
        ci = 0
        for c in C:
            self.N_doc = len(D)  # number of documents in D
            self.N_c = documents_per_class(c, D)  # number of documents from D in category c
            self.logprior[ci] = math.log(self.N_c / self.N_doc)  # prior probs for each category c
            self.V = define_vocabulary(D)  # generate vocabulary V over the set of documents D
            self.bigdoc[ci] = populate_bigdoc(c, D)
            for w in self.V:
                self.loglikelihood[(w, c)] = self.loglikelihood.get((w, c), 0) + math.log(util.count(w, self.bigdoc[ci]) + 1 / sum(util.count(w_, c) + 1 for w_ in self.V))
            ci += 1

        return self.logprior, self.loglikelihood, self.V

    def test_naive_bayes(self, testdoc, logprior, loglikelihood, C, V):
        sum_ = []
        ci = 0
        for c in C:
            sum_.append(logprior[ci])
            for w in testdoc:
                word = w
                if word in V:
                    sum_[ci] += loglikelihood.get((word, c), 0) + 1
            ci += 1
        if sum_.index(max(sum_)) == 0:
            best_c = 'neg'
        else:
            best_c = 'pos'

        return best_c

    def populate_categories_list(self, C):
        categories = []
        for c in C:
            if c not in self.C:
                self.C.append(c)
            categories.append(c)

        return categories


categories = ['neg', 'pos']
vocabulary_filepath = 'input-files/vocabulary/imdb.vocab'
vocabulary_vectorized = prep.vectorize(vocabulary_filepath)[0]

""" JUNK/TEST/FAKE PROGRAM """
classifier_JUNK = NBClassifier()

corpora_raw_JUNK = ['input-files/raw/test_JUNK/neg', 'input-files/raw/test_JUNK/pos', 'input-files/raw/train_JUNK/neg', 'input-files/raw/train_JUNK/pos']
for corpus in corpora_raw_JUNK:  # Normalize all corpora
    for file in os.listdir(corpus):
        file_path = os.path.join(corpus, file)
        if os.path.isfile(file_path):
            prep.normalize(file_path)

training_corpora_vectorized_JUNK = []
training_corpora_processed_JUNK = ['input-files/processed/train_JUNK/neg', 'input-files/processed/train_JUNK/pos']
for corpus in training_corpora_processed_JUNK:
    for file in os.listdir(corpus):
        file_path = os.path.join(corpus, file)
        if os.path.isfile(file_path):
            training_corpora_vectorized_JUNK.append(prep.vectorize(file_path))

testing_corpora_vectorized_JUNK = []
testing_corpora_processed_JUNK = ['input-files/processed/test_JUNK/neg', 'input-files/processed/test_JUNK/pos']

filenames = []
i = 0
for corpus in testing_corpora_processed_JUNK:
    for file in os.listdir(corpus):
        filenames.append(file)
        file_path = os.path.join(corpus, file)
        if os.path.isfile(file_path):
            testing_corpora_vectorized_JUNK.append(prep.vectorize(file_path)[0])
        i += 1

training_vector_JUNK = classifier_JUNK.train_naive_bayes(training_corpora_vectorized_JUNK, categories)
prep.write_back(training_vector_JUNK, 'output-files/vector.txt')
prob_prior_JUNK = training_vector_JUNK[0]
prep.write_back(prob_prior_JUNK, 'output-files/probs_prior.txt')
prob_JUNK = training_vector_JUNK[1]
prep.write_back(prob_JUNK, 'output-files/probs.txt')
f = 0
for tscv_JUNK in testing_corpora_vectorized_JUNK:
    print("Document " + filenames[f] + ": " + classifier_JUNK.test_naive_bayes(tscv_JUNK, prob_prior_JUNK, prob_JUNK, categories, vocabulary_vectorized))
    f += 1
""" JUNK/TEST/FAKE PROGRAM """

""" 'REAL' PROGRAM """
# classifier = NBClassifier()
#
# corpora_raw = ['input-files/raw/test/neg', 'input-files/raw/test/pos', 'input-files/raw/train/neg', 'input-files/raw/train/pos']
# for corpus in corpora_raw:  # Normalize all corpora
#     for file in os.listdir(corpus):
#         file_path = os.path.join(corpus, file)
#         if os.path.isfile(file_path):
#             prep.normalize(file_path)
#
# training_corpora_vectorized = []
# training_corpora_processed = ['input-files/processed/train/neg', 'input-files/processed/train/pos']
# for corpus in training_corpora_processed:
#     for file in os.listdir(corpus):
#         file_path = os.path.join(corpus, file)
#         if os.path.isfile(file_path):
#             training_corpora_vectorized.append(prep.vectorize(file_path))
#
# testing_corpora_vectorized = []
# testing_corpora_processed = ['input-files/processed/test/neg', 'input-files/processed/test/pos']
#
# filenames = []
# i = 0
# for corpus in testing_corpora_processed:
#     for file in os.listdir(corpus):
#         filenames.append(file)
#         file_path = os.path.join(corpus, file)
#         if os.path.isfile(file_path):
#             testing_corpora_vectorized.append(prep.vectorize(file_path)[0])
#         i += 1
#
# training_vector = classifier.train_naive_bayes(training_corpora_vectorized, categories)
# prep.write_back(training_vector, 'output-files/vector.txt')
# prob_prior = training_vector[0]
# prep.write_back(prob_prior, 'output-files/probs_prior.txt')
# prob = training_vector[1]
# prep.write_back(prob, 'output-files/probs.txt')
# f = 0
# for tscv in testing_corpora_vectorized:
#     print("Document " + filenames[f] + ": " + classifier.test_naive_bayes(tscv, prob_prior, prob, categories, vocabulary_vectorized))
#     f += 1

""" 'REAL' PROGRAM """
