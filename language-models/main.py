import math
from models.unigram_model import UnigramModel


def normalize(corpora: iter):
    """
    T   =   the current corpus being normalized
    S   =   sentences:  the set of all sentences in the corpus
    s   =   sentence:   a single, non-normalized sentence
    w   =   word:   a single word token

    ** NOTE: This function assumes that the text file(s) in `corpora` has/have already been tokenized **

    Takes a list of one or more text files (corpora) and normalizes them:
    All text becomes lower-case, and '<s>' and '</s>' are added as leading and trailing sentinels, respectively, to each line.
    Each of the normalized text files in `corpora` are then written back to files of the same name and put in the `input_data/processed/` directory.

    :param corpora: an iterable set of text files (corpora) to be normalized
    :return: None
    """
    for corpus in corpora:
        with open(file=corpus, mode='r', encoding='utf-8') as T:  # read in from file
            S = []  # S: list
            for s in T:
                s = s.strip().lower()
                if not s.startswith('<s>'):
                    s = '<s> ' + s
                if not s.endswith('</s>'):
                    s = s + ' </s>'
                s = s.split()  # .split() returns a list of substrings (words) in `s`
                S.append(s)
        corpus_processed = corpus.replace('raw', 'processed')
        with open(file=corpus_processed, mode='w', encoding='utf-8') as T:  # write to new file
            for w in S:
                T.write(' '.join(w) + '\n')


def mark_singletons(normalized_corpus):
    V = {}  # V: dict for counting the occurrences of each word in the normalized corpus
    with open(file=normalized_corpus, mode='r', encoding='utf-8') as T:
        for s in T:
            W = s.strip().split()
            for w in W:
                V[w] = V.get(w, 0) + 1
    # Replace singletons in the normalized corpus with the `<unk>` token:
    with open(file=normalized_corpus, mode='r', encoding='utf-8') as T:
        S = []  # S: list for storing sentences
        for s in T:
            S.append(s.strip())
        for i, s in enumerate(S):
            W = s.split()
            S[i] = ' '.join([w if w == '<s>' else '<unk>' if V[w] == 1 else w for w in W])
    # Write the modified train sentences back to the normalized corpus
    with open(file=normalized_corpus, mode='w', encoding='utf-8') as T:
        for s in S:
            T.write(s + '\n')


if __name__ == '__main__':
    # Normalize the corpora:
    normalize(['input_data/raw/train.txt', 'input_data/raw/test.txt'])
    normalize(['input_data/raw/garbage.txt'])

    # used for frequencies of words occurring prior to marking unknowns:
    uni_unmarked_training = UnigramModel()
    uni_unmarked_testing = UnigramModel()

    # used for frequencies of words occurring after marking unknowns:
    uni_marked_training = UnigramModel()
    uni_marked_testing = UnigramModel()

    uni_unmarked_training.train('input_data/processed/train.txt')  # train the designated unigram model on the normalized corpus with no `<unk>` tokens

    vocab_size_unmarked_training = uni_unmarked_training.unigram_count()
    total_words_unmarked_training = uni_unmarked_training.token_count()

    print("\nVocabulary size (before marking singletons):", vocab_size_unmarked_training)
    print("Total word tokens (before marking singletons):", total_words_unmarked_training)

    print("\nQuestion 1: How many word types (unique words) are there in the training corpus? Include `<unk>` and `</s>`. Do not include `<s>`.")
    print("Marking `<unk>` tokens in (normalized) training text...")
    mark_singletons('input_data/processed/train.txt')
    uni_marked_training.train('input_data/processed/train.txt')
    vocab_size_marked_training = uni_marked_training.unigram_count()
    print("Vocabulary size (after marking singletons):", vocab_size_marked_training)

    print("\nQuestion 2: How many word tokens are there in the training corpus? Do not include `<s>`.")
    total_words_marked_training = uni_marked_training.token_count()
    print("Total word tokens (after marking singletons):", total_words_marked_training, '\n')

