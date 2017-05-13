import tensorflow as tf
import numpy as np
import time

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
        sess = self.sess

        # shape of y : [None, seq_length], shape of logits : [None, seq_length, vocab_size]
        # convert shape of logits to match y: from one-hot to integer vocab index
        ######### implement here

        #########

        # calculate loss and cost
        loss = sequence_loss(logits=logits, targets=y)
        cost = tf.reduce_sum(loss) / batch_size / self.seq_length
        train_cost_summary = tf.summary.scalar('train_cost', cost)
        test_cost_summary = tf.summary.scalar('test_cost', cost)

        # load trainable variables
        tvars = tf.trainable_variables()
        # apply gradient clipping for stable learning of RNN
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), clip_norm=5.)

        # define optimizer
        optimizer = train.AdamOptimizer(learning_rate=lr, beta1=decay)
        # apply gradients of cost to trainable variables
        train_op = optimizer.apply_gradients(cost, tvars)

        # writer for tensorboard
        writer = tf.summary.FileWriter('./tmp/char_rnn', sess.graph)

        # training loop
        import time
        # split data into training/test set
        seed = np.random.randint(0, 100, 1)
        inputs = sess.run(tf.random_shuffle(inputs, seed))
        outputs = sess.run(tf.random_shuffle(outputs,seed))
        num_valid = validation_split * inputs.shape[0]
        x_train = inputs[0:num_valid]
        y_train = outputs[0:num_valid]
        x_test = inputs[num_valid:]
        y_test = outputs[num_valid:]

        # calculate the number of mini batch
        total_batch = int(len(x_train) // batch_size)
        # start training iteration
        print("Training started...")
        # start timer to measure the time taken for a step
        start_time = time.time()
        # for each epoch
        for epoch in range(epochs):
            # shuffle training set
            seed = np.random.randint(0, 100, 1)
            x_train = sess.run(tf.random_shuffle(x_train, seed))
            y_train = sess.run(tf.random_shuffle(y_train, seed))
            # for each mini-batch
            for step in range(total_batch):
                start_ind = step * batch_size
                end_ind = (step + 1) * batch_size
                train_feed = {x: x_train[start_ind:end_ind], y: y_train[start_ind:end_ind]}
                write_train_cost, _ = self.sess.run([train_cost_summary, train_op], feed_dict=train_feed)
                if step % 100 == 0:
                    writer.add_summary(write_train_cost, (epoch * total_batch + step))
                    # test
                    test_feed = {x: x_test[start_ind:end_ind], y: y_test[start_ind:end_ind]}
                    write_test_cost = self.sess.run(test_cost_summary, feed_dict=test_feed)
                    writer.add_summary(write_test_cost, (epoch * total_batch + step))
                    print('\repoch : %d, batch : %d/%d data, step_time : %.2fsec, train_loss : %4f, test_loss : %4f'
                          % ((epoch + 1), (step + 1) * batch_size, x_train.shape[0], (time.time() - start_time),
                             loss.eval(train_feed), loss.eval(test_feed)))
                    start_time = time.time()

        # save checkpoints every few epochs
        ######### implement here

        #########
        print("Optimization Finished!")
        tf.summary.FileWriter.close(writer)
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