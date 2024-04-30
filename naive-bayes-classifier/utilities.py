import math


def tf(t, d):
    tf = 0
    if count(t, d) > 0:
        tf = 1 + math.log10(tf)

    return tf


def df(t, D):
    df = 0
    for d in D:
        df += d.split.count(t)

    return df


def idf(t, D):
    return math.log10(len(D) / df(t, D))


def cf(t, D):
    cf = 0
    for d in D:
        if t in d:
            cf += 1

    return cf


def count(t, d):
    return d.count(t)


def load_words_with_count(file_path):
    word_dict = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            word = line.strip()
            if word in word_dict:
                word_dict[word] += 1
            else:
                word_dict[word] = 1

    return word_dict
