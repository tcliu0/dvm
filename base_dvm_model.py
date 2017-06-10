import os
import logging
import tensorflow as tf
#import matplotlib.pyplot as plt

from util import Progbar, get_minibatches

class BaseDVMSystem(object):
    def __init__(self, flags):
        # Save commandline parameters
        self.flags = flags

        # Set up placeholder tokens
        self.i1_placeholder = tf.placeholder(tf.float32, (None, self.flags.height, self.flags.width, self.flags.channels))
        self.i2_placeholder = tf.placeholder(tf.float32, (None, self.flags.height, self.flags.width, self.flags.channels))
        self.gt_placeholder = tf.placeholder(tf.float32, (None, self.flags.height, self.flags.width, self.flags.channels))
        if self.flags.use_depth:
            self.dm_placeholder = tf.placeholder(tf.float32, (None, self.flags.height, self.flags.width))

        with tf.variable_scope('dvm'):
            self.setup_system()
            self.setup_loss()

        optimizer = tf.train.AdamOptimizer(self.flags.learning_rate)
        gradients, v = zip(*optimizer.compute_gradients(self.loss))
        self.norm = tf.global_norm(gradients)
        self.train_op = optimizer.apply_gradients(zip(gradients, v))
        self.saver = tf.train.Saver(max_to_keep=10)

    def setup_system(self):
        return NotImplemented

    def setup_loss(self):
        with tf.variable_scope('loss'):
            norm = tf.reduce_sum(tf.pow(self.R - self.gt_placeholder, 2.0), axis=3)
            self.loss = tf.reduce_mean(norm)
            if self.flags.use_depth:
                self.r_loss = self.loss
                self.dm_loss = tf.reduce_mean(tf.pow(self.DM - self.dm_placeholder, 2.0))
                self.loss += self.dm_loss

    def optimize(self, session, dataset, epoch):
        input_feed = {}
        output_feed = [self.train_op, self.loss, self.norm]

        train, loader = dataset

        total_loss = 0.

        prog = Progbar(target=(len(train) - 1) / self.flags.batch_size + 1)
        for i, batch in enumerate(get_minibatches(train, self.flags.batch_size)):
            input_feed[self.i1_placeholder] = [loader(b[0]) for b in batch]
            input_feed[self.i2_placeholder] = [loader(b[1]) for b in batch]
            input_feed[self.gt_placeholder] = [loader(b[2]) for b in batch]
            if self.flags.use_depth:
                input_feed[self.dm_placeholder] = [loader(b[3]) for b in batch]
          
            _, loss, norm = session.run(output_feed, input_feed)
            prog.update(i+1, [("train loss", loss), ("norm", norm)])
            total_loss += loss

        return total_loss

    def predict(self, session, test_data):
        input_feed = {}
        output_feed = [self.R]

        test, loader = test_data

        input_feed[self.i1_placeholder] = [loader(test[0])]
        input_feed[self.i2_placeholder] = [loader(test[1])]

        result = session.run(output_feed, input_feed)[0]
        if self.flags.use_depth:
            output_feed = [self.DM]
            depth = session.run(output_feed, input_feed)[0]
            result = (result, depth)
        return result

    def evaluate(self, session, dataset):
        input_feed = {}
        output_feed = self.loss

        test, loader = dataset

        total_loss = 0.

        prog = Progbar(target=(len(test) - 1) / self.flags.batch_size + 1)
        for i, batch in enumerate(get_minibatches(test, self.flags.batch_size, shuffle=False)):
            input_feed[self.i1_placeholder] = [loader(b[0]) for b in batch]
            input_feed[self.i2_placeholder] = [loader(b[1]) for b in batch]
            input_feed[self.gt_placeholder] = [loader(b[2]) for b in batch]
            if self.flags.use_depth:
                input_feed[self.dm_placeholder] = [loader(b[3]) for b in batch]
            
            total_loss += session.run(output_feed, input_feed) * len(batch)
            prog.update(i+1, exact=[("total loss", total_loss)])

        return total_loss

    def train(self, session, dataset, start_epoch=0):
        train_dir = os.path.join(self.flags.train_dir, self.model_name)
        for epoch in range(start_epoch, self.flags.epochs):
            logging.info("Epoch %d out of %d", epoch + 1, self.flags.epochs)
            self.optimize(session, dataset, epoch)
            
            self.saver.save(session, '%s/%s.ckpt' % (train_dir, self.model_name), global_step=epoch)
            logging.info("Saving model in %s", train_dir)
