import os
import json
import tensorflow as tf

from dvm_model import DVMSystem
from dataset import load_dataset

import logging

logging.basicConfig(level=logging.INFO)

tf.app.flags.DEFINE_integer('height', 256, 'Image height in pixels.')
tf.app.flags.DEFINE_integer('width', 256, 'Image width in pixels.')
tf.app.flags.DEFINE_integer('channels', 3, 'Number of channels in image.')
tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate.')
tf.app.flags.DEFINE_integer('epochs', 10, 'Number of epochs to train.')
tf.app.flags.DEFINE_integer("batch_size", 100, "Batch size to use during training.")
tf.app.flags.DEFINE_string('dataset', 'caltech3d', 'Name of dataset to train on.')
tf.app.flags.DEFINE_bool('use_depth', False, 'Train network for depth reconstruction')
tf.app.flags.DEFINE_string("train_dir", "train", "Training directory to save the model parameters (default: ./train).")
tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")

FLAGS = tf.app.flags.FLAGS

def initialize_model(session, model, train_dir):
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
        epoch = int(ckpt.model_checkpoint_path.split('-')[-1]) + 1
    else:
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
        epoch = 0
    return model, epoch

def main(_):
    if not os.path.exists(FLAGS.train_dir):
        os.makedirs(FLAGS.train_dir)

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    file_handler = logging.FileHandler(os.path.join(FLAGS.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)

    train, test, loader = load_dataset(FLAGS.dataset)

    dvm = DVMSystem(FLAGS)

    train_dir = os.path.join(FLAGS.train_dir, dvm.model_name)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    print(vars(FLAGS))
    with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    with tf.Session() as sess:
        _, epoch = initialize_model(sess, dvm, train_dir)
        
        dvm.train(sess, (train, loader), epoch)

if __name__ == '__main__':
    tf.app.run()    
