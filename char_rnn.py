import tensorflow as tf
import numpy as np
import time
import os

class model:
    def __init__(self, width, depth, is_training, seq_length, vocab_size, sess):
        self.width = width
        self.depth = depth
        self.is_training = is_training
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.sess = sess
        self.inputs_placeholder = tf.placeholder(tf.float32,
                                                 shape=[None, self.seq_length, self.vocab_size])
        self.targets_placeholder = tf.placeholder(tf.int64,
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
        lstm_cell = rnn.LayerNormBasicLSTMCell(num_units=self.width, reuse=tf.get_variable_scope().reuse)

        # currently multiRNNCell seems to have a bug
        # lstm_layers = rnn.MultiRNNCell([lstm_cell for _ in range(self.depth)])

        # just one layer instead
        lstm_layers = lstm_cell

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
            b = tf.get_variable('b', [self.vocab_size],
                                initializer=tf.constant_initializer(0.))  # zero bias for start, just to be safe
        # currently logits is list of 2d tensors
        logits = [tf.matmul(output, W) + b for output in lstm_outputs]  # list of (batch size, vocab_size), for each time step
        # convert to 3d tensors, by stacking the list
        # sequence_loss in seq2seq api assumes 3d tensors of [batch_size, seq_length, vocab_size]
        logits = tf.stack(logits, axis=1)

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

        # build the model
        # shape of y : [None, seq_length], shape of x and logits : [None, seq_length, vocab_size]
        x = self.inputs_placeholder
        y = self.targets_placeholder
        logits = self.build(batch_size=batch_size)
        sess = self.sess

        # we assume unbiased estimates of the loss per time step
        loss_weights = tf.constant(np.ones((batch_size, self.seq_length)), dtype=tf.float32)

        # calculate loss
        # sequence_loss returns scalar loss
        loss = sequence_loss(logits=logits, targets=y, weights=loss_weights)
        # cost = tf.reduce_sum(loss) / batch_size / self.seq_length
        cost = loss
        train_cost_summary = tf.summary.scalar('train_cost', cost)
        test_cost_summary = tf.summary.scalar('test_cost', cost)

        # load trainable variables
        tvars = tf.trainable_variables()
        # apply gradient clipping for stable learning of RNN
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), clip_norm=5.)

        # define optimizer
        optimizer = train.AdamOptimizer(learning_rate=lr, beta1=decay)
        # apply gradients of cost to trainable variables
        train_op = optimizer.apply_gradients(zip(grads, tvars))

        # writer for tensorboard
        writer = tf.summary.FileWriter('./tmp/char_rnn', sess.graph)

        # training loop
        import time
        # split data into training/test set
        seed = np.random.randint(0, 100, 1)
        feed = list(zip(inputs, outputs))
        np.random.shuffle(feed)
        inputs_feed, outputs_feed = zip(*feed)
        inputs_feed = np.array(inputs_feed)
        outputs_feed = np.array(outputs_feed)

        # split the training set to train and valid set
        num_valid = np.int32(validation_split * inputs_feed.shape[0])
        x_train = inputs_feed[0:num_valid]
        y_train = outputs_feed[0:num_valid]
        x_test = inputs_feed[num_valid:]
        y_test = outputs_feed[num_valid:]

        # calculate the number of mini batch
        total_batch = int(len(x_train) // batch_size)
        # start training iteration
        print("Training started...")

        # initialize variables
        self.sess.run(tf.global_variables_initializer())

        # define saver
        self.saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=1)
        # load checkpoints
        self.load_checkpoint()

        # start timer to measure the time taken for a step
        start_time = time.time()
        # for each epoch
        for epoch in range(epochs):
            # shuffle training set
            feed = list(zip(x_train, y_train))
            np.random.shuffle(feed)
            x_train, y_train = zip(*feed)
            x_train = np.array(x_train)
            y_train = np.array(y_train)

            # for each mini-batch
            for step in range(total_batch):
                # calculate index for current batch
                start_ind = step * batch_size
                end_ind = (step + 1) * batch_size
                # generate feed_dict
                train_feed = {x: x_train[start_ind:end_ind], y: y_train[start_ind:end_ind]}
                # train step
                write_train_cost, _ = self.sess.run([train_cost_summary, train_op], feed_dict=train_feed)
                # write cost every several steps
                if step % 10 == 0:
                    writer.add_summary(write_train_cost, (epoch * total_batch + step))
                    # test
                    test_feed = {x: x_test[start_ind:end_ind], y: y_test[start_ind:end_ind]}
                    write_test_cost = self.sess.run(test_cost_summary, feed_dict=test_feed)
                    writer.add_summary(write_test_cost, (epoch * total_batch + step))
                    print('\repoch : %d, batch : %d/%d data, step_time : %.2fsec, train_loss : %4f, test_loss : %4f'
                          % ((epoch + 1), (step + 1) * batch_size, x_train.shape[0], (time.time() - start_time),
                             loss.eval(train_feed), loss.eval(test_feed)))
                    start_time = time.time()

            self.save_checkpoint()

        print("Optimization Finished!")
        tf.summary.FileWriter.close(writer)
        return


    def generate_sample(self, vocab, dic_vocab, num_chars, primer):
        """
        generate samples from the model
        :param num_chars: number of characters for sampling
        :param primer: string for initial inputs of RNN
        :return: None
        """
        # load checkpoints
        #self.load_checkpoint()

        # build the model
        x = self.inputs_placeholder
        logits = self.build(batch_size=1)

        # generate character with vocab

        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            result = int(np.searchsorted(t, np.random.rand(1) * s))
            return (result)

        ret = primer
        char = primer[-1]
        for n in range(num_chars):
            data = np.zeros((1, 1))
            data[0, 0] = vocab[char]

            logits_r = self.sess.run([logits], feed={x: data})
            p = tf.nn.softmax(logits_r)[0]

            sample = weighted_pick(p)
            pred = vocab[sample]
            ret += pred
            char = pred
        print(ret)


    def load_checkpoint(self):
        ckpt = tf.train.get_checkpoint_state('save')
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess,
                               tf.train.latest_checkpoint('save'))
        return


    def save_checkpoint(self):
        checkpoint_path = os.path.join('save', 'model.ckpt')
        self.saver.save(self.sess, checkpoint_path)
        print("model saved to {}".format(checkpoint_path))
        return