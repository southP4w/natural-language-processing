import math

from models.bigram_model import BigramModel
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


def mark_singletons(normalized_corpus: iter):
    V = {}  # V: dict for counting the occurrences of each word in the normalized corpus
    with open(file=normalized_corpus, mode='r', encoding='utf-8') as T:
        for s in T:
            W = s.strip().split()
            for w in W:
                V[w] = V.get(w, 0) + 1
    with open(file=normalized_corpus, mode='r', encoding='utf-8') as T:  # replace singletons in the normalized corpus with the `<unk>` token
        S = []  # S: list for storing sentences
        for s in T:
            S.append(s.strip())
        for i, s in enumerate(S):
            W = s.split()
            S[i] = ' '.join([w if w == '<s>' else '<unk>' if V[w] == 1 else w for w in W])
    with open(file=normalized_corpus, mode='w', encoding='utf-8') as T:  # write the modified train sentences back to the normalized corpus
        for s in S:
            T.write(s + '\n')


def missing_unigrams_ratio(test_corpus: iter, unigrams: dict) -> str:
    missing_unigrams = 0
    total_unigrams = 0
    with open(file=test_corpus, mode='r', encoding='utf-8') as T:
        for s in T:
            for w in s.split():
                if w != '<s>':
                    total_unigrams += 1
                elif w not in unigrams:
                    missing_unigrams += 1

    return '%.4f' % (missing_unigrams / total_unigrams * 100)


def missing_uni_token_ratio(test_corpus: iter, unigrams: dict) -> str:
    missing_tokens = 0
    total_tokens = 0
    with open(file=test_corpus, mode='r', encoding='utf-8') as T:
        for s in T:
            for w in s.split():
                if w != '<s>':
                    total_tokens += 1
                    if w not in unigrams:
                        missing_tokens += 1

    return '%.4f' % (missing_tokens / total_tokens * 100)


def missing_bigrams_ratio(test_corpus: iter, bigrams: dict) -> str:
    missing_bigrams = 0
    total_bigrams = 0
    with open(file=test_corpus, mode='r', encoding='utf-8') as T:
        for s in T:
            W = s.split()
            for i in range(len(W) - 1):
                total_bigrams += 1
                bigram = (W[i], W[i + 1])
                if bigram not in bigrams:
                    missing_bigrams += 1

    return '%.4f' % (missing_bigrams / total_bigrams * 100)


def missing_bi_token_ratio(test_corpus: iter, bigrams: iter) -> str:
    missing_bigrams = 0
    total_bigrams = 0
    with open(file=test_corpus, mode='r', encoding='utf-8') as T:
        for s in T:
            W = s.split()
            for i in range(len(W) - 1):
                if W[i] != '<s>':
                    total_bigrams += 1
                    bigram = (W[i], W[i + 1])
                    if bigram not in bigrams:
                        missing_bigrams += 1

    return '%.4f' % (missing_bigrams / total_bigrams * 100)


if __name__ == '__main__':
    normalize(['input_data/raw/train.txt', 'input_data/raw/test.txt', 'input_data/raw/garbage.txt'])  # normalize the corpora

    # used for frequencies of words occurring prior to marking unknowns:
    uni_unmarked_training = UnigramModel()
    uni_unmarked_testing = UnigramModel()  # used strictly for counting tokens/unigrams in the testing corpus

    # used for frequencies of words occurring after marking unknowns:
    uni_marked_training = UnigramModel()
    uni_marked_testing = UnigramModel()  # used strictly for counting tokens/unigrams in the testing corpus

    uni_unmarked_training.train('input_data/processed/train.txt')  # train the designated unigram model on the normalized corpus with no `<unk>` tokens
    uni_unmarked_testing.train('input_data/processed/test.txt')

    training_vocab_size_unmarked = uni_unmarked_training.get_unigram_count()
    training_token_count_unmarked = uni_unmarked_training.get_token_count()
    testing_vocab_size_unmarked = uni_unmarked_testing.get_unigram_count()
    testing_token_count_unmarked = uni_unmarked_testing.get_token_count()

    print("\nBefore marking singletons:")
    print("'train.txt' vocabulary size:", training_vocab_size_unmarked)
    print("'train.txt' total word tokens:", training_token_count_unmarked)
    print("'test.txt' vocabulary size:", testing_vocab_size_unmarked)
    print("'test.txt' total word tokens:", testing_token_count_unmarked)

    print("\nQuestion 1: How many word types (unique words) are there in the training corpus? Include `<unk>` and `</s>`. Do not include `<s>`.")
    mark_singletons('input_data/processed/train.txt')
    uni_marked_training.train('input_data/processed/train.txt')
    training_vocab_size_marked = uni_marked_training.get_unigram_count()
    print("Vocabulary size (after marking singletons):", training_vocab_size_marked)

    print("\nQuestion 2: How many word tokens are there in the training corpus? Do not include `<s>`.")
    print("Total word tokens (after marking singletons):", uni_marked_training.get_token_count())

    print("\nQuestion 3:")
    print("What percentage of word tokens and word types in the test corpus did not occur in "
          "training (before mapping the unknown words to `<unk>` in training and test data)?")
    print("Include `</s>` in your calculations. Do not include `<s>`.")
    print(f"Original 'train.txt' vocabulary size = {training_vocab_size_unmarked}")
    print(f"Marked `<unk>`'s vocabulary size = {training_vocab_size_marked}")
    print(f"Ratio (unigram types):", missing_unigrams_ratio('input_data/processed/test.txt', uni_unmarked_training.V))
    print(f"Ratio (unigram tokens):", missing_uni_token_ratio('input_data/processed/test.txt', uni_unmarked_training.W))

    print("\nQuestion 4:")
    print("Now replace singletons in the training data with `<unk>` symbol and map words (in the test corpus) not "
          "observed in training to `<unk>`.)")
    print("What percentage of bigrams (bigram types and bigram "
          "tokens) in the test corpus did not occur in training (treat `<unk>` as a regular token that "
          "has been observed).")
    print("Include `</s>`. Do not include `<s>`.")
    bi_marked_training = BigramModel()
    bi_marked_training.train('input_data/processed/train.txt')
    print(f"Ratio (bigram types):", missing_bigrams_ratio('input_data/processed/test.txt', bi_marked_training.V_1))
    print(f"Ratio (bigram tokens):", missing_bi_token_ratio('input_data/processed/test.txt', bi_marked_training.W))

    print("\nQuestion 5:")
    print("Compute the log probability of the sentence 'I look forward to hearing your reply .' under the three models.")
    print("(Ignore capitalization and pad each sentence as described above).")
    print("Please list all of the parameters required to compute the probabilities and show the complete calculation.")
    print("Which of the parameters have zero values under each model?")
    print("Use log_2 in your calculations.")
    print("Map words not observed in training to the `<unk>` token.\n")
    print("UNIGRAM:")
    print(f"\nTotal (Unigram): {uni_marked_training.compute_probability_log("I look forward to hearing your reply .")}")
    print("\nBIGRAM:")
    print(f"\nTotal (Bigram): {bi_marked_training.compute_probability_log("I look forward to hearing your reply .")}")
    print(f"\nBIGRAM (SMOOTHED):")
    print(f"\nTotal (Smoothed Bigram): {bi_marked_training.compute_probability_smoothed("I look forward to hearing your reply .")}")

    print("\nQuestion 6:")
    print("Compute the perplexity of 'I look forward to hearing your reply .' under each of the models.")
    print(f"Unigram: {uni_marked_training.compute_perplexity_of_sentence("I look forward to hearing your reply .")}")
    print(f"Bigram: {bi_marked_training.compute_perplexity_of_sentence("I look forward to hearing your reply .")}")
    print(f"Bigram (smoothed): {bi_marked_training.compute_perplexity_smoothed_of_sentence("I look forward to hearing your reply .")}")

    print("\nQuestion 7:")
    print("Compute the perplexity of the entire test corpus under each of the models.")
    print("Dicsuss the differences in the results obtained.")
    print(f"Unigram: {uni_marked_training.compute_perplexity_of_corpus('input_data/processed/test.txt')}")
    print(f"Bigram: {bi_marked_training.compute_perplexity_of_corpus('input_data/processed/test.txt')}")
    print(f"Bigram: {bi_marked_training.compute_perplexity_smoothed_of_corpus('input_data/processed/test.txt')}")
