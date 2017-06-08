import tensorflow as tf

from base_dvm_model import BaseDVMSystem

class DVMBaselineSystem(BaseDVMSystem):
    def __init__(self, flags):
        super(DVMBaselineSystem, self).__init__(flags)
        self.model_name = 'dvm_baseline'

    def setup_system(self):
        self.tensor_dict = {}

        # Fuse input images
        i = tf.concat((self.i1_placeholder, self.i2_placeholder), 3)

        W1 = tf.get_variable('W1', (9, 9, 2*self.flags.channels, 32), initializer=tf.contrib.layers.xavier_initializer())
        W2 = tf.get_variable('W2', (7, 7, 32, 64), initializer=tf.contrib.layers.xavier_initializer())
        W3 = tf.get_variable('W3', (5, 5, 64, 128), initializer=tf.contrib.layers.xavier_initializer())
        W4 = tf.get_variable('W4', (3, 3, 128, 256), initializer=tf.contrib.layers.xavier_initializer())
        #W5 = tf.get_variable('W5', (1, 1, 256, 512), initializer=tf.contrib.layers.xavier_initializer())
        #W6 = tf.get_variable('W6', (1, 1, 512, 512), initializer=tf.contrib.layers.xavier_initializer())
        W7 = tf.get_variable('W7', (1, 1, 256, self.flags.channels), initializer=tf.contrib.layers.xavier_initializer())

        b1 = tf.get_variable('b1', (32), initializer=tf.constant_initializer(0.1))
        b2 = tf.get_variable('b2', (64), initializer=tf.constant_initializer(0.1))
        b3 = tf.get_variable('b3', (128), initializer=tf.constant_initializer(0.1))
        b4 = tf.get_variable('b4', (256), initializer=tf.constant_initializer(0.1))
        #b5 = tf.get_variable('b5', (512), initializer=tf.constant_initializer(0.1))
        #b6 = tf.get_variable('b6', (512), initializer=tf.constant_initializer(0.1))
        b7 = tf.get_variable('b7', (self.flags.channels), initializer=tf.constant_initializer(0.1))
        
        h1 = tf.nn.relu(tf.nn.conv2d(i, W1, strides=[1, 1, 1, 1], padding='SAME') + b1)
        h2 = tf.nn.relu(tf.nn.conv2d(h1, W2, strides=[1, 1, 1, 1], padding='SAME') + b2)
        h3 = tf.nn.relu(tf.nn.conv2d(h2, W3, strides=[1, 1, 1, 1], padding='SAME') + b3)
        h4 = tf.nn.relu(tf.nn.conv2d(h3, W4, strides=[1, 1, 1, 1], padding='SAME') + b4)
        #h5 = tf.nn.relu(tf.nn.conv2d(h4, W5, strides=[1, 1, 1, 1], padding='SAME') + b5)
        #h6 = tf.nn.relu(tf.nn.conv2d(h5, W6, strides=[1, 1, 1, 1], padding='SAME') + b6)
        h7 = tf.nn.relu(tf.nn.conv2d(h4, W7, strides=[1, 1, 1, 1], padding='SAME') + b7)
        
        self.R = h7
