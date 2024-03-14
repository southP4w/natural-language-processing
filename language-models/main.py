from models.unigram_model import UnigramModel


def normalize(corpora: iter):
    """
    T   =   the current corpus (text file) being normalized
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


if __name__ == '__main__':
    normalize(['input_data/raw/train.txt', 'input_data/raw/test.txt'])
    u = UnigramModel()
    u.train('input_data/processed/train.txt')
    # print(u.unigram_count())
    # print(len(u.W))
