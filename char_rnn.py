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
                                                  shape=[None, self.seq_length])
        return


    def build(self, batch_size):
        """
        build the model
        should return target of the same size
        when training, it is [batch_size, seq_length]
        when sampling, it is [1, 1]
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
        :param inputs: input data for training, has shape of inputs_placeholder
        :param outputs: output data for training, has shape of targets_placeholder
        :param batch_size: batch size
        :param epochs: pre-specified max epochs
        :param lr: learning rate
        :param decay: factor for lr decay
        :param validation_split: portion of validation data of range [0, 1]
        :return: None
        """

        from tensorflow.contrib.seq2seq import sequence_loss
        from tensorflow import train
        # load checkpoints
        ######### implement here
        checkpoint = self.load_checkpoint()
        # apply checkpoint to the model
        #########

        # build the model
        x = self.inputs_placeholder
        y = self.targets_placeholder
        logits = self.build(batch_size=batch_size)

        # shape of y : [None, seq_length], shape of logits : [None, seq_length, vocab_size]
        # convert shape of logits to match y: from one-hot to integer vocab index
        ######### implement here

        #########

        # calculate loss and cost
        loss = sequence_loss(logits=logits, targets=y)
        cost = tf.reduce_sum(loss) / batch_size / self.seq_length

        # load trainable variables
        tvars = tf.trainable_variables()
        # apply gradient clipping for stable learning of RNN
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), clip_norm=5.)

        # define optimizer
        optimizer = train.AdamOptimizer(learning_rate=lr, beta1=decay)
        # apply gradients of cost to trainable variables
        train_op = optimizer.apply_gradients(cost, tvars)

        # training loop
        # save checkpoints every few epochs
        ######### implement here

        #########

        return


    def generate_sample(self, num_chars, primer):
        """
        generate samples from the model
        :param num_chars: number of characters for sampling
        :param primer: string for initial inputs of RNN
        :return: None
        """
        ######### implement here
        checkpoint = self.load_checkpoint()
        # apply checkpoint to the model
        #########

        # build the model
        x = self.inputs_placeholder
        logits = model.build(batch_size=1)

        # generate character with vocab
        return


    def load_checkpoint(self):
        checkpoint = None
        return checkpoint


    def save_checkpoint(self):
        return