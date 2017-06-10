import os
import json
import random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc

#from dvm_baseline_model import DVMBaselineSystem as DVMSystem
from dvm_model import DVMSystem
from dataset import load_dataset
from util import Progbar

import logging

logging.basicConfig(level=logging.INFO)

tf.app.flags.DEFINE_integer('height', 256, 'Image height in pixels.')
tf.app.flags.DEFINE_integer('width', 256, 'Image width in pixels.')
tf.app.flags.DEFINE_integer('channels', 3, 'Number of channels in image.')
tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate.')
tf.app.flags.DEFINE_integer('epochs', 10, 'Number of epochs to train.')
tf.app.flags.DEFINE_integer("batch_size", 100, "Batch size to use during training.")
tf.app.flags.DEFINE_string('dataset', 'caltech3d', 'Name of dataset to train on.')
tf.app.flags.DEFINE_bool('use_depth', False, 'Train network for depth reconstruction.')
tf.app.flags.DEFINE_string("train_dir", "train", "Training directory to save the model parameters (default: ./train).")
tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")

FLAGS = tf.app.flags.FLAGS

def initialize_model(session, model, train_dir):
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
    return model

def main(_):
    if not os.path.exists(FLAGS.train_dir):
        os.makedirs(FLAGS.train_dir)

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    file_handler = logging.FileHandler(os.path.join(FLAGS.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)

    dataset = load_dataset(FLAGS.dataset)
    if len(dataset) == 4:
        train, test, val, loader = dataset
    else:
        train, test, loader = dataset

    dvm = DVMSystem(FLAGS)
    
    train_dir = os.path.join(FLAGS.train_dir, dvm.model_name)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    print(vars(FLAGS))
    with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    with tf.Session() as sess:
        initialize_model(sess, dvm, train_dir)
        
        total_loss = dvm.evaluate(sess, (test, loader))
        print total_loss

        # for test_data in test[12345:12346]:
        #     if FLAGS.use_depth:
        #         (result, ), (depth, ) = dvm.predict(sess, (test_data, loader))
        #     else:
        #         result = dvm.predict(sess, (test_data, loader))[0]

        #     input_feed = {dvm.i1_placeholder: [loader(test_data[0])],
        #                   dvm.i2_placeholder: [loader(test_data[1])],
        #                   dvm.gt_placeholder: [loader(test_data[2])],
        #                   dvm.dm_placeholder: [loader(test_data[3])]}

        #     i1 = sess.run(dvm.i1_placeholder, input_feed)[0]
        #     i2 = sess.run(dvm.i2_placeholder, input_feed)[0]
        #     gt = sess.run(dvm.gt_placeholder, input_feed)[0]
        #     gtdm = sess.run(dvm.dm_placeholder, input_feed)[0]
        #     r1 = sess.run(dvm.tensor_dict['R1'], input_feed)[0]
        #     r2 = sess.run(dvm.tensor_dict['R2'], input_feed)[0]
        #     r1p1 = sess.run(dvm.vm.tensor_dict['R1(P1)'], input_feed)[0]
        #     r2p2 = sess.run(dvm.vm.tensor_dict['R2(P2)'], input_feed)[0]
        #     c = sess.run(dvm.tensor_dict['C'], input_feed)[0].squeeze().reshape((-1, ))
        #     m = sess.run(dvm.tensor_dict['M'], input_feed)[0].squeeze()
        #     r = sess.run(dvm.R, input_feed)[0]
        #     dm = sess.run(dvm.DM, input_feed)[0]
        #     plt.imshow(r)
        #     plt.show()

        #     max_c = []
        #     rows = []
        #     while True:
        #         argmax = np.argmax(c)
        #         maxval = c[argmax]
        #         c[argmax] = 0

        #         cont = False
        #         y = argmax / 256
        #         for i in range(y-5, y+5):
        #             if i in rows:
        #                 cont = True
        #         if cont:
        #             continue
                
        #         rows.append(argmax/256)
        #         max_c.append((argmax, maxval))
        #         if len(max_c) == 10:
        #             break

        #     r1r2 = np.concatenate((r1, r2), axis=1)
        #     plt.imshow(r1r2)
        #     for argmax, c in max_c:
        #         y = argmax / 256
        #         x = argmax % 256
        #         plt.plot([x + c, x - c + 256], [y, y], 'x-')

        #     err = np.linalg.norm(gt - r, axis=2)

        #     misc.imsave('dvm_sn_i1.png', i1)
        #     misc.imsave('dvm_sn_i2.png', i2)
        #     misc.imsave('dvm_sn_gt.png', gt)
        #     misc.imsave('dvm_sn_r1.png', r1)
        #     misc.imsave('dvm_sn_r2.png', r2)
        #     misc.imsave('dvm_sn_r1p1.png', r1p1)
        #     misc.imsave('dvm_sn_r2p2.png', r2p2)
        #     plt.savefig('dvm_sn_c.png')
        #     misc.imsave('dvm_sn_m.png', m)
        #     misc.imsave('dvm_sn_r.png', r)
        #     misc.imsave('dvm_sn_err.png', err)
        #     misc.imsave('dm.png', dm)
        #     misc.imsave('gtdm.png', gtdm)
        #     misc.imsave('dm_err.png', np.abs(err))

        #     r = sess.run(dvm.R, input_feed)[0]
        #     misc.imsave('baseline_ct3d_r.png', r)

if __name__ == '__main__':
    tf.app.run()    
