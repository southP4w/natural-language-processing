# v =   vocabulary                  =   set of all words in a language


def preprocess_data(training_corpus, testing_corpus):
    def handle_unknowns(tokens: list, types: dict, include_unknowns: bool):
        if include_unknowns:
            for i, phrase in enumerate(tokens):
                tokens[i] = ['<unk>' if types[w] == 1 and w not in ('<s>', '</s>')
                             else w for w in phrase
                             ]
        else:
            for i, phrase in enumerate(tokens):
                tokens[i] = ['' if types[w] == 1 and w not in ('<s>', '</s>')
                             else w for w in phrase
                             ]

    u = {}  # dict for storing all unique words (types) in the corpus (no repeats)
    with open(file=training_corpus, mode='r') as f:  # f = the current file being read from/written to
        n = []  # list for storing all preprocessed words (tokens) in the corpus (has repeats)
        for c in f:  # c = current line (not processed);     For the current line in the file,
            c = c.strip().lower()  # strip leading and trailing whitespace in `c` and make `c` all lowercase.
            if not c.startswith('<s>'):  # If no leading '<s>' in this line,
                c = '<s> ' + c  # add it.
            if not c.endswith('</s>'):  # If no trailing '</s>' in this line,
                c = c + ' </s>'  # add it.
            p = c.split()  # p = preprocessed `c`
            n.append(p)  # add the now preprocessed current line's tokens to `n`
            for w in p:  # w = a single word;    For each word in the current processed line,
                print()     # NEED TO FINISH
                # u[w] = u.get(w, 0) + 1  # update the counts of each word in `p` in the list `n`.

    handle_unknowns(tokens=n, types=u, include_unknowns=True)


if __name__ == '__main__':
    print()
