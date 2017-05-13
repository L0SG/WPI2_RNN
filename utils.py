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
    tf.one_hot


    return inputs_embed, targets_embed





import codecs
import os
import collections
from six.moves import cPickle
import numpy as np


class TextLoader():
    def __init__(self, data_dir, batch_size, seq_length, encoding='utf-8'):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.encoding = encoding

        input_file = os.path.join(data_dir, "input.txt")
        vocab_file = os.path.join(data_dir, "vocab.pkl")
        tensor_file = os.path.join(data_dir, "data.npy")

        if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
            print("reading text file")
            self.preprocess(input_file, vocab_file, tensor_file)
        else:
            print("loading preprocessed files")
            self.load_preprocessed(vocab_file, tensor_file)
        self.create_batches()
        self.reset_batch_pointer()

    def preprocess(self, input_file, vocab_file, tensor_file):
        with codecs.open(input_file, "r", encoding=self.encoding) as f:
            data = f.read()
        counter = collections.Counter(data)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        self.chars, _ = zip(*count_pairs)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.chars, f)
        self.tensor = np.array(list(map(self.vocab.get, data)))
        np.save(tensor_file, self.tensor)

    def load_preprocessed(self, vocab_file, tensor_file):
        with open(vocab_file, 'rb') as f:
            self.chars = cPickle.load(f)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.tensor = np.load(tensor_file)
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length))

    def create_batches(self):
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length))

        # When the data (tensor) is too small,
        # let's give them a better error message
        if self.num_batches == 0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
        xdata = self.tensor
        ydata = np.copy(self.tensor)
        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]
        self.x_batches = np.split(xdata.reshape(self.batch_size, -1),
                                  self.num_batches, 1)
        self.y_batches = np.split(ydata.reshape(self.batch_size, -1),
                                  self.num_batches, 1)

    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0