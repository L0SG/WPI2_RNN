import tensorflow as tf
import numpy as np

class model:
    def __init__(self, width, depth, is_training, seq_length, embed_size, sess):
        self.width = width
        self.depth = depth
        self.is_training = is_training
        self.seq_length = seq_length
        self.embed_size = embed_size
        self.sess = sess
        self.inputs_placeholder = tf.placeholder(tf.float32,
                                                 shape=[None, self.seq_length, self.embed_size])
        self.targets_placeholder = tf.placeholder(tf.float32,
                                                  shape=[None, self.seq_length, self.embed_size])
        return


    def build(self):
        """
        build the model
        should return target of the same size
        when training, it is [batch_size, seq_length, embed_size]
        when sampling, it is [1, 1, embed_size]
        :return: final state output for targets
        softmax will be calculated at train & generate_sample
        """
        logits = None
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
        logits = model.build()
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