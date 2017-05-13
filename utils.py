import tensorflow as tf
from keras.models import Sequential
from keras.layers import Embedding
import numpy as np
import codecs
import os
import collections


def load_data(file_path, seq_length, sess):
    """
    load the text file and generate tokenized inputs and targets
    given data array of size n, y[i] = x[i-1] for i=1:n
    each x[i] and y[i] is tokenized integer,
    with vocab of english character, punctuations, and eof token
    :param file_path: the file path of .txt file
    :return: inputs x and targets y, with the size of [len(data) (# of characters)]
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
    x_token = np.array(list(map(vocab.get, data)))
    y_token = np.copy(x_token)
    # make y[i] = x[i-1]
    y_token[:-1] = x_token[1:]
    y_token[-1] = x_token[0]

    # reshape (-1, seq_length)
    x_reshape = np.reshape(x_token[: len(x_token) // seq_length * seq_length], (-1, seq_length))
    y_reshape = np.reshape(y_token[: len(y_token) // seq_length * seq_length], (-1, seq_length))

    # one-hot vector
    x = sess.run(tf.one_hot(x_reshape, len(vocab), axis=-1))
    y = sess.run(tf.one_hot(y_reshape, len(vocab), axis=-1))

    return x, y, vocab


def preprocess(inputs, targets, vocab, batch_size, seq_length, embed_size):
    """
    preprocess the tokenized inputs and targets
    to generate embed vectors from vocab
    embedding size is determined by embed_size
    :return: embedded matrix of inputs and targets
    with the sizeof [-1, seq_length, embed_size]
    """
    inputs_reshaped = np.reshape(inputs[0:len(inputs)//seq_length * seq_length], (-1, seq_length))
    targets_reshaped = np.reshape(targets[0:len(targets) // seq_length * seq_length], (-1, seq_length))

    # embedding
    model = Sequential()
    model.add(Embedding(len(vocab), embed_size, input_length=seq_length))
    # the model will take as input an integer matrix of size (batch_size, seq_length).
    # now model.output_shape == (None, 10, 64), where None is the batch dimension.
    model.compile('rmsprop', 'mse')

    inputs_embed = model.predict(inputs_reshaped)
    targets_embed = model.predict(targets_reshaped)

    return inputs_embed, targets_embed