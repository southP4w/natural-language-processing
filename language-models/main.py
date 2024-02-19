# v =   vocabulary                  =   set of all words in a language


def preprocess_data(train_file, test_file, include_unknowns):
    word_counts = {}

    with open(file=train_file, mode='r', encoding='utf-8') as file:
        train_sentences = []
        for line in file:
            line = line.strip().lower()
            # Ensure <s> and </s> are not already present:
            if not line.startswith('<s>'):
                line = '<s> ' + line
            if not line.endswith('</s>'):
                line = line + ' </s>'
            words = line.split()
            train_sentences.append(words)
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1

    if include_unknowns:
        # Process the words in train.txt
        for i, sentence in enumerate(train_sentences):
            train_sentences[i] = ['<unk>' if word_counts[word] == 1 and word not in ('<s>', '</s>') else word for word in sentence]
    else:
        for i, sentence in enumerate(train_sentences):
            train_sentences[i] = ['' if word_counts[word] == 1 and word not in ('<s>', '</s>') else word for word in sentence]

    # Write the processed sentences back to train.txt
    with open(file=train_file, mode='w', encoding='utf-8') as file:
        for sentence in train_sentences:
            file.write(' '.join(sentence) + '\n')

    # Process test file
    with open(file=test_file, mode='r', encoding='utf-8') as file:
        test_sentences = []
        for line in file:
            line = line.strip().lower()
            if not line.startswith('<s>'):
                line = '<s> ' + line
            if not line.endswith('</s>'):
                line = line + ' </s>'
            words = line.split()
            test_sentences.append(words)

    if include_unknowns:
        for i, sentence in enumerate(test_sentences):
            test_sentences[i] = ['<unk>' if word not in word_counts or word_counts[word] == 1 else word for word in sentence]
    else:
        for i, sentence in enumerate(test_sentences):
            test_sentences[i] = [word for word in sentence]

    with open(file=test_file, mode='w', encoding='utf-8') as file:
        for sentence in test_sentences:
            file.write(' '.join(sentence) + '\n')


def map_singletons(train_file, test_file):
    word_counts = {}  # Dictionary to count the occurrences of each word in train.txt

    # Process train file to get word counts
    with open(file=train_file, mode='r', encoding='utf-8') as file:
        for line in file:
            words = line.strip().split()
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1

    # Process train file again to replace singletons with <unk>
    with open(file=train_file, mode='r', encoding='utf-8') as file:
        train_sentences = [line.strip() for line in file]
        for i, sentence in enumerate(train_sentences):
            words = sentence.split()
            train_sentences[i] = ' '.join(['<unk>' if word_counts[word] == 1 else word for word in words])

    # Write the modified train sentences back to train.txt
    with open(file=train_file, mode='w', encoding='utf-8') as file:
        for sentence in train_sentences:
            file.write(sentence + '\n')

    # Process test file to map unobserved words to <unk>
    with open(file=test_file, mode='r', encoding='utf-8') as file:
        test_sentences = [line.strip() for line in file]
        for i, sentence in enumerate(test_sentences):
            words = sentence.split()
            test_sentences[i] = ' '.join(['<unk>' if word not in word_counts else word for word in words])

    # Write the modified test sentences back to test.tx
    with open(file=test_file, mode='w', encoding='utf-8') as file:
        for sentence in test_sentences:
            file.write(sentence + '\n')


if __name__ == '__main__':
    print()
