import tensorflow as tf
import numpy as np


def load_data(file_path):
    """
    load the text file and generate tokenized inputs and targets
    given data array of size n, y[i] = x[i-1] for i=1:n
    each x[i] and y[i] is tokenized integer,
    with vocab of english character, punctuations, and eof token
    :param file_path: the file path of .txt file
    :return: inputs x and targets y, with the size of [len(charcters)-1, len(seq_length)]
    """
    x, y, vocab = None
    return x, y, vocab


def preprocess(inputs, targets, vocab, batch_size, seq_length, embed_size):
    """
    preprocess the tokenized inputs and targets
    to generate embed vectors from vocab
    embedding size is determined by embed_size
    :return: embedded matrix of inputs and targets
    with the sizeof [batch_size, seq_length, embed_size]
    """
    inputs_embed, targets_embed = None
    return inputs_embed, targets_embed