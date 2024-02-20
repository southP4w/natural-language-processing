# v =   vocabulary  =   set of all word types in a language

def normalize(corpora: iter):
    """
    u   =   word types:     the set of all unique words in the corpus (has no repeats)
    n   =   word tokens:    the set of ALL words in the corpus (has repeats)
    w   =   a single word token
    f   =   the current `corpus` (text file) being normalized
    c   =   the current line in `f`
    p   =   normalized `c`

    ** NOTE: This function assumes that the text file(s) in `corpora` has/have already been tokenized **

    Takes a list of one or more text files (corpora) and normalizes them:
    All text becomes lower-case, and '<s>' and '</s>' are added as leading and trailing sentinels, respectively, to each line.
    Each of the normalized text files in `corpora` are then written back to their respective files.

    :param corpora: an iterable set of text files (corpora) to be normalized
    :return: None
    """
    for corpus in corpora:
        u = {}  # u: dict
        with open(file=corpus, mode='r', encoding='utf-8') as f:  # read in
            n = []  # n: list
            for c in f:
                c = c.strip().lower()
                if not c.startswith('<s>'):
                    c = '<s> ' + c
                if not c.endswith('</s>'):
                    c = c + ' </s>'
                p = c.split()
                n.append(p)
                for w_ in p:
                    u[w_] = u.get(w_, 0) + 1
        with open(file=corpus, mode='w', encoding='utf-8') as f:  # write back
            for w in n:
                f.write(' '.join(w) + '\n')


if __name__ == '__main__':
    normalize(['input_data/raw/test-Fall2023.txt', 'input_data/raw/train-Fall2023.txt'])
