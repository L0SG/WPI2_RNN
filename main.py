import tensorflow as tf
import numpy as np
import utils
from char_rnn import model

# mode switch: True if training phase, False if sampling phase
is_training = True

# hyperparameters for model
width = 128
depth = 3
seq_length = 100
if not is_training:
    seq_length = 1
embed_size = 50

# hyperparameters for training
batch_size = 32
if not is_training:
    batch_size = 1
epochs = 100
learning_rate = 1e-4
weight_decay = 0.99
validation_split = 0.1
dataset = 'shakespeare.txt'

# hyperparameters for sampling
num_chars = 200

sess = tf.InteractiveSession()

if is_training:
    """
    # load the text data
    # text is a huge array containing characters
    text_in, text_out, vocab = utils.load_data(dataset)

    # generate x and y from the text
    x, y = utils.preprocess(inputs=text_in, targets=text_out, vocab=vocab,
                            batch_size=batch_size, seq_length=seq_length, embed_size=embed_size)
    """

    # calculate vocab_size
    vocab_size = 10
    x, y = None, None
    # build the char-rnn model
    rnn_model = model(width=width, depth=depth, is_training=is_training, seq_length=seq_length,
                      embed_size=embed_size, vocab_size=vocab_size, sess=sess)

    # load checkpoint if exists
    rnn_model.load_checkpoint()

    # train the model
    rnn_model.train(inputs=x, outputs=y, batch_size=batch_size,
                    epochs=epochs, lr=learning_rate, decay=weight_decay,
                    validation_split=validation_split)

    # save checkpoint of the model
    rnn_model.save_checkpoint()

else:
    # build the char-rnn model
    rnn_model = model(width=width, depth=depth, is_training=is_training,
                      seq_length=seq_length, embed_size=embed_size, sess=sess)

    # load checkpoint
    rnn_model.load_checkpoint()

    # generate texts
    rnn_model.generate_sample(num_chars=num_chars, primer='I am')

