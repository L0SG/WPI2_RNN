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
        :param inputs:
        :param outputs:
        :param batch_size:
        :param epochs:
        :param lr:
        :param decay:
        :param validation_split:
        :return:
        """

        """
        # make config for tf session
        config = tf.ConfigProto()
        # enable to grow the memory usage as is needed by process
        config.gpu_options.allow_growth = True
        # limit maximum memory usage to 0.3
        config.gpu_options.per_process_gpu_memory_fraction = 0.3
        # to print which devices your operations and tensors are assigned to, change this to True
        config.log_device_placement = False
        # automatically choose an supported device to run the operations when the specified one doesn't exist
        config.allow_soft_placement = True
        # create a session with the config specified above
        sess = tf.Session(config=config)
        """

        # initialize all tf variable
        # make operation to initialize all tf variables
        var_init = tf.global_variables_initializer()
        self.sess.run(var_init)

        # create file writer and an event file in tmp directory
        writer = tf.summary.FileWriter('./tmp/char_rnn', self.sess.graph)
        saver = tf.train.Saver()

        # load checkpoints
        if tf.train.latest_checkpoint('./ckpt') is not None:
            saver.restore(self.sess, tf.train.latest_checkpoint(checkpoint_dir='./ckpt'))
            print('model restored!')

        # build the model
        print('building LSTM model..')
        logits = model.build()
        print('building LSTM model complete!')

        # calculate loss and summary to write a log file
        with tf.variable_scope('cost'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=outputs))
            train_loss_summary = tf.summary.scalar('train_loss', loss)
            test_loss_summary = tf.summary.scalar('test_loss', loss)
        # optimizer with decaying learning rate
        with tf.variable_scope('optimizer'):
            step = tf.Variable(0, trainable=False)
            rate = tf.train.exponential_decay(lr, step, 1, decay)
            optimizer = tf.train.AdamOptimizer(rate).minimize(loss, global_step=step)

        # calculate the number of mini batch
        total_batch = int(len(inputs) // batch_size)
        # start training iteration
        print("Training started...")
        # start timer to measure the time taken for a step
        start_time = time.time()
        # for each epoch
        for epoch in range(epochs):
            # np.random.shuffle(x)
            # for each mini-batch
            for step in range(total_batch):
                start_ind = step * batch_size
                end_ind = (step + 1) * batch_size
                train_feed = {self.inputs_placeholder: inputs[start_ind:end_ind],
                              self.targets_placeholder: outputs[start_ind:end_ind]}
                write_train_loss, _ = self.sess.run([train_loss_summary, optimizer],feed_dict=train_feed)
                if step % 100 == 0:
                    writer.add_summary(write_train_loss, (epoch * total_batch + step))
                    # test
                    test_feed = {self.inputs_placeholder: inputs[start_ind:end_ind],
                                 self.targets_placeholder: outputs[start_ind:end_ind]}
                    write_test_loss = self.sess.run(test_loss_summary, feed_dict=test_feed)
                    writer.add_summary(write_test_loss, (epoch * total_batch + step))
                    # print(tf.nn.softmax(logits).eval(train_feed)) # check output of the model
                    print('\repoch : %d, batch : %d/%d data, step_time : %.2fsec, train_loss : %4f, test_loss : %4f'
                        % ((epoch+1), (step+1) * batch_size, inputs.shape[0], (time.time() - start_time),
                           loss.eval(train_feed), loss.eval(test_feed)))
                    start_time = time.time()
            # save checkpoints every few epochs
            if epoch % 1 == 0:
                save_path = saver.save(self.sess,'./ckpt/char_rnn.ckpt',global_step=epoch)
                print('model is saved to %s' %save_path)

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