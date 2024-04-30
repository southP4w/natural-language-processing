import os
import re as regex


def normalize(corpus_raw_filepath):
    E = ["mr.", "ms.", "mrs.", "dr.", "esq.", "col.", "maj.", "pvt.", "cpl.", "lt.", "gen.", "prof.", "jr.", "sr."]
    with open(file=corpus_raw_filepath, mode='r', encoding='utf-8') as T:
        t = T.read().lower()  # t <- lowercased corpus text T
    R = []  # set of processed sentences/lines
    s = []  # current sentence/line
    W = t.split()  # W <- t in list form
    for w in W:
        s.append(w)
        if w.endswith('.') and not any(w.endswith(e) for e in E):
            R.append(' '.join(s))
            s = []
    if s:
        R.append(' '.join(s))
    corpus_processed_filepath = corpus_raw_filepath.replace('raw', 'processed')
    os.makedirs(os.path.dirname(corpus_processed_filepath), exist_ok=True)
    with open(corpus_processed_filepath, 'w', encoding='utf-8') as T:
        for r in R:  # for each processed line r in R:
            r = regex.sub(r"([.,;:'\"?/()\[\]{}=+\-`~\\!@#$%^|&*<>])", r" \1 ", r)
            r = regex.sub(r"\s+", " ", r).strip()
            T.write(r + ' ')

    return corpus_processed_filepath


def write_back(vector, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for element in vector:
            file.write(f"{element}\n")


def vectorize(corpus_processed_filepath):
    category = os.path.basename(os.path.dirname(corpus_processed_filepath))
    with open(corpus_processed_filepath, 'r', encoding='utf-8') as T:
        text = T.read()
    W = text.split()

    return W, category
