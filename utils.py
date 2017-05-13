import tensorflow as tf
import numpy as np
import codecs
import os
import collections


def load_data(file_path):
    """
    load the text file and generate tokenized inputs and targets
    given data array of size n, y[i] = x[i-1] for i=1:n
    each x[i] and y[i] is tokenized integer,
    with vocab of english character, punctuations, and eof token
    :param file_path: the file path of .txt file
    :return: inputs x and targets y, with the size of [len(charcters)-1, len(seq_length)]
    """
    # load txt file
    input_file = os.path.join(file_path)
    with codecs.open(input_file, "r", encoding='utf-8') as f:
        data = f.read()

    # make vocabulary
    # count the number of each character in dataset
    count_char = collections.Counter(data)
    # sort counting list in descending order
    sort_char = sorted(count_char.items(), key=lambda x: -x[1])
    # arrange every characters in dataset
    characters, _ = zip(*sort_char)
    # make vocab
    vocab = dict(zip(characters, range(len(characters))))

    # tokenizing
    x = np.array(list(data))
    y = np.copy(x)
    # make y[i] = x[i-1]
    y[:-1] = x[1:]
    y[-1] = x[0]

    return x, y, vocab


def preprocess(inputs, targets, vocab, batch_size, seq_length, embed_size):
    """
    preprocess the tokenized inputs and targets
    to generate embed vectors from vocab
    embedding size is determined by embed_size
    :return: embedded matrix of inputs and targets
    with the sizeof [batch_size, seq_length, embed_size]
    """
    # change token to vocab index integer
    inputs_int = map(vocab.get, inputs)
    targets_int = map(vocab.get, targets)

    # reshape
    inputs_reshape = np.reshape(inputs_int, (batch_size,seq_length))
    # word embedding



    return inputs_embed, targets_embed