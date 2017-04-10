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

# hyperparameters for training
batch_size = 50
epochs = 100
learning_rate = 1e-4
weight_decay = 0.99
validation_split = 0.1

# hyperparameters for sampling
num_chars = 200


# load the text data
# text is a huge array containing characters
text = utils.load_data('shakespeare.txt')

# generate x and y from the text
x, y = utils.preprocess(inputs=text, seq_length=seq_length)

# build the char-rnn model
rnn_model = model(width=width, depth=depth, seq_length=seq_length)

# load checkpoint if exists
rnn_model.load_checkpoint()

if is_training:
    # train the model
    rnn_model.train(inputs=x, outputs=y, batch_size=batch_size,
                    epochs=epochs, lr=learning_rate, decay=weight_decay,
                    validation_split=validation_split)

    # save checkpoint of the model
    rnn_model.save_checkpoint()

else:
    # generate texts
    rnn_model.generate_sample(num_chars=num_chars)

