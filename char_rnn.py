import tensorflow as tf
import numpy as np

class model:
    def __init__(self, width, depth, is_training, seq_length, embed_size, vocab_size, sess):
        self.width = width
        self.depth = depth
        self.is_training = is_training
        self.seq_length = seq_length
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.sess = sess
        self.inputs_placeholder = tf.placeholder(tf.float32,
                                                 shape=[None, self.seq_length, self.embed_size])
        self.targets_placeholder = tf.placeholder(tf.float32,
                                                  shape=[None, self.seq_length, self.embed_size])
        return


    def build(self, batch_size):
        """
        build the model
        should return target of the same size
        when training, it is [batch_size, seq_length, embed_size]
        when sampling, it is [1, 1, embed_size]
        :return: final state output for targets
        softmax will be calculated at train & generate_sample
        """
        from tensorflow.contrib import rnn

        # input placeholder
        x = self.inputs_placeholder
        # define standard lstm networks
        lstm_cell = rnn.LayerNormBasicLSTMCell(num_units=self.width)
        lstm_layers = rnn.MultiRNNCell([lstm_cell] * self.depth)

        # define the empty outputs list for appending the output of each time step
        lstm_outputs = []
        # initial lstm state is zero
        state = lstm_layers.zero_state(batch_size=batch_size, dtype=tf.float32)

        # lstm loop (inefficient) for studying purpose
        # using RNN API (tf.nn.rnn) is better
        with tf.variable_scope('lstm') as scope_lstm:
            for time_step in range(self.seq_length):
                # create variables (lstm_output, state) at the initial step, and reuse this after
                if time_step > 0:
                    scope_lstm.reuse_variables()
                # feed the inputs of [all batches, one time step, all embeddings], and states
                (lstm_output, state) = lstm_layers(x[:, time_step, :], state)
                # append the output, state is not needed out of the loop
                lstm_outputs.append(lstm_output)

        # calculate logits of each step from lstm_outputs

        with tf.variable_scope('logits'):
            W = tf.get_variable('W', [self.width, self.vocab_size])
            b = tf.get_variable('b', [self.vocab_size], initializer=tf.constant_initializer(0.)) # zero bias for start, just to be safe
        logits = [tf.matmul(output, W) + b for output in lstm_outputs] # list of (batch size, vocab_size), for each time step

        return logits

    def train(self, inputs, outputs, batch_size, epochs, lr, decay, validation_split):
        """
        train the model
        calculate loss and apply gradients
        load checkpoint if exists
        save checkpoint every few epochs
        :param inputs:
        :param outputs:
        :param batch_size:
        :param epochs:
        :param lr:
        :param decay:
        :param validation_split:
        :return:
        """
        # load checkpoints

        # build the model
        logits = self.build(batch_size=batch_size)
        # calculate loss and apply grads

        # save checkpoints every few epochs
        return


    def generate_sample(self, num_chars, primer):
        """
        generate samples from the model
        :param num_chars:
        :param primer:
        :return:
        """
        # load checkpoints

        # build the model
        logits = model.build()
        # generate character with vocab
        return


    def load_checkpoint(self):
        return


    def save_checkpoint(self):
        return