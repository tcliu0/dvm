import numpy as np
import tensorflow as tf

from base_dvm_model import BaseDVMSystem

class Rectification(object):
    def __init__(self, flags):
        self.flags = flags
        self.tensor_dict = {}

        eye = np.reshape(np.eye(3, dtype=np.float32), (9, ))
        self.eye = np.concatenate((eye, eye))
       
        height, width = (self.flags.height, self.flags.width)
        points = np.reshape(np.mgrid[0:height, 0:width], (2, height, width, 1))
        points = np.concatenate((points[0], points[1], np.ones_like(points[0])), axis=2)
        points = np.reshape(points, (-1, 3))
        self.points = tf.constant(points, dtype=tf.float32)

    def rectification_early_fusion(self, i1, i2, scope='rect_early'):
        def scaled_xavier(scale=0.05):
            def xavier(*args, **kwargs):
                return scale * tf.contrib.layers.xavier_initializer()(*args, **kwargs)
            return xavier
        
        with tf.variable_scope(scope):
            W_rc1 = tf.get_variable('W_rc1', (9, 9, 2*self.flags.channels, 32), initializer=tf.contrib.layers.xavier_initializer())
            b_rc1 = tf.get_variable('b_rc1', (32), initializer=tf.constant_initializer(0.01))
            W_rc2 = tf.get_variable('W_rc2', (7, 7, 32, 64), initializer=tf.contrib.layers.xavier_initializer())
            b_rc2 = tf.get_variable('b_rc2', (64), initializer=tf.constant_initializer(0.01))
            W_rc3 = tf.get_variable('W_rc3', (5, 5, 64, 128), initializer=tf.contrib.layers.xavier_initializer())
            b_rc3 = tf.get_variable('b_rc3', (128), initializer=tf.constant_initializer(0.01))
            W_rc4 = tf.get_variable('W_rc4', (3, 3, 128, 256), initializer=tf.contrib.layers.xavier_initializer())
            b_rc4 = tf.get_variable('b_rc4', (256), initializer=tf.constant_initializer(0.01))
            W_rc5 = tf.get_variable('W_rc5', (3, 3, 256, 512), initializer=tf.contrib.layers.xavier_initializer())
            b_rc5 = tf.get_variable('b_rc5', (512), initializer=tf.constant_initializer(0.01))
            W_rc6 = tf.get_variable('W_rc6', (1, 1, 512, 512), initializer=tf.contrib.layers.xavier_initializer())
            b_rc6 = tf.get_variable('b_rc6', (512), initializer=tf.constant_initializer(0.01))
            W_rc7 = tf.get_variable('W_rc7', (1, 1, 512, 512), initializer=tf.contrib.layers.xavier_initializer())
            b_rc7 = tf.get_variable('b_rc7', (512), initializer=tf.constant_initializer(0.01))
            W_rc8 = tf.get_variable('W_rc8', (1, 1, 512, 18), initializer=scaled_xavier())
            b_rc8 = tf.get_variable('b_rc8', initializer=self.eye)

            Si = tf.concat((i1, i2), 3)
            RC1 = tf.nn.relu(tf.nn.conv2d(Si, W_rc1, strides=[1, 2, 2, 1], padding='SAME') + b_rc1)
            RP1 = tf.nn.max_pool(RC1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
            RC2 = tf.nn.relu(tf.nn.conv2d(RP1, W_rc2, strides=[1, 1, 1, 1], padding='SAME') + b_rc2)
            RP2 = tf.nn.max_pool(RC2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
            RC3 = tf.nn.relu(tf.nn.conv2d(RP2, W_rc3, strides=[1, 1, 1, 1], padding='SAME') + b_rc3)
            RP3 = tf.nn.max_pool(RC3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
            RC4 = tf.nn.relu(tf.nn.conv2d(RP3, W_rc4, strides=[1, 1, 1, 1], padding='SAME') + b_rc4)
            RP4 = tf.nn.max_pool(RC4, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
            RC5 = tf.nn.relu(tf.nn.conv2d(RP4, W_rc5, strides=[1, 1, 1, 1], padding='SAME') + b_rc5)
            RP5 = tf.nn.avg_pool(RC5, ksize=[1, RC5.get_shape()[1], RC5.get_shape()[2], 1], strides=[1, 1, 1, 1], padding='VALID')
            RC6 = tf.nn.relu(tf.nn.conv2d(RP5, W_rc6, strides=[1, 1, 1, 1], padding='SAME') + b_rc6)
            RC7 = tf.nn.relu(tf.nn.conv2d(RC6, W_rc7, strides=[1, 1, 1, 1], padding='SAME') + b_rc7)
            RC8 = tf.nn.conv2d(RC7, W_rc8, strides=[1, 1, 1, 1], padding='SAME') + b_rc8

            H1, H2 = tf.split(tf.reshape(RC8, [-1, 18]), 2, axis=1)
            return H1, H2

    def rectification_late_fusion(self, i1, i2, scope='rect_late'):
        def conv_tower_pre_fusion(im):
            W_rc1 = tf.get_variable('W_rc1', (9, 9, self.flags.channels, 32), initializer=tf.contrib.layers.xavier_initializer())
            b_rc1 = tf.get_variable('b_rc1', (32), initializer=tf.constant_initializer(0.01))
            W_rc2 = tf.get_variable('W_rc2', (7, 7, 32, 64), initializer=tf.contrib.layers.xavier_initializer())
            b_rc2 = tf.get_variable('b_rc2', (64), initializer=tf.constant_initializer(0.01))
            W_rc3 = tf.get_variable('W_rc3', (5, 5, 64, 128), initializer=tf.contrib.layers.xavier_initializer())
            b_rc3 = tf.get_variable('b_rc3', (128), initializer=tf.constant_initializer(0.01))
            W_rc4 = tf.get_variable('W_rc4', (3, 3, 128, 256), initializer=tf.contrib.layers.xavier_initializer())
            b_rc4 = tf.get_variable('b_rc4', (256), initializer=tf.constant_initializer(0.01))
            W_rc5 = tf.get_variable('W_rc5', (3, 3, 256, 256), initializer=tf.contrib.layers.xavier_initializer())
            b_rc5 = tf.get_variable('b_rc5', (256), initializer=tf.constant_initializer(0.01))            

            RC1 = tf.nn.relu(tf.nn.conv2d(im, W_rc1, strides=[1, 2, 2, 1], padding='SAME') + b_rc1)
            RP1 = tf.nn.max_pool(RC1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
            RC2 = tf.nn.relu(tf.nn.conv2d(RP1, W_rc2, strides=[1, 1, 1, 1], padding='SAME') + b_rc2)
            RP2 = tf.nn.max_pool(RC2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
            RC3 = tf.nn.relu(tf.nn.conv2d(RP2, W_rc3, strides=[1, 1, 1, 1], padding='SAME') + b_rc3)
            RP3 = tf.nn.max_pool(RC3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
            RC4 = tf.nn.relu(tf.nn.conv2d(RP3, W_rc4, strides=[1, 1, 1, 1], padding='SAME') + b_rc4)
            RP4 = tf.nn.max_pool(RC4, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
            RC5 = tf.nn.relu(tf.nn.conv2d(RP4, W_rc5, strides=[1, 1, 1, 1], padding='SAME') + b_rc5)
            RP5 = tf.nn.avg_pool(RC5, ksize=[1, RC5.get_shape()[1], RC5.get_shape()[2], 1], strides=[1, 1, 1, 1], padding='VALID')
            return RP5

        def conv_tower_post_fusion(i1, i2):
            RP5 = tf.concat((i1, i2), 3)
            
            W_rc6 = tf.get_variable('W_rc6', (1, 1, 512, 512), initializer=tf.contrib.layers.xavier_initializer())
            b_rc6 = tf.get_variable('b_rc6', (512), initializer=tf.constant_initializer(0.01))
            W_rc7 = tf.get_variable('W_rc7', (1, 1, 512, 512), initializer=tf.contrib.layers.xavier_initializer())
            b_rc7 = tf.get_variable('b_rc7', (512), initializer=tf.constant_initializer(0.01))
            W_rc8 = tf.get_variable('W_rc8', (1, 1, 512, 9), initializer=tf.contrib.layers.xavier_initializer())
            b_rc8 = tf.get_variable('b_rc8', (9), initializer=tf.constant_initializer(0.01))

            RC6 = tf.nn.relu(tf.nn.conv2d(RP5, W_rc6, strides=[1, 1, 1, 1], padding='SAME') + b_rc6)
            RC7 = tf.nn.relu(tf.nn.conv2d(RC6, W_rc7, strides=[1, 1, 1, 1], padding='SAME') + b_rc7)
            RC8 = tf.nn.conv2d(RC7, W_rc8, strides=[1, 1, 1, 1], padding='SAME') + b_rc8
            return RC8

        with tf.variable_scope(scope):
            RP5_1 = conv_tower_pre_fusion(i1)
            tf.get_variable_scope().reuse_variables()
            RP5_2 = conv_tower_pre_fusion(i2)

        with tf.variable_scope(scope):
            H1 = conv_tower_post_fusion(RP5_1, RP5_2)
            tf.get_variable_scope().reuse_variables()
            H2 = conv_tower_post_fusion(RP5_2, RP5_1)

        return H1, H2

    def rectification(self, im, h, scope='rectification'):
        def get_values(im, p):
            x_max = tf.shape(im)[2] - 1
            y_max = tf.shape(im)[1] - 1

            x_coord = p[:, :, 0:1]
            y_coord = p[:, :, 1:2]
            x_coord = tf.maximum(tf.minimum(x_coord, x_max), 0)
            y_coord = tf.maximum(tf.minimum(y_coord, y_max), 0)

            p = tf.concat((x_coord, y_coord), axis=2)
            
            idx = tf.expand_dims(tf.range(start=0, limit=tf.shape(p)[0], delta=1), axis=1)
            idx = tf.tile(idx, [1, self.flags.height * self.flags.width])
            idx = tf.concat((tf.expand_dims(idx, axis=2), p), axis=2)
            
            values = tf.gather_nd(im, idx)
            return values

        with tf.variable_scope(scope):        
            hmat = tf.reshape(h, (-1, 3, 3))
            transformed_points = tf.einsum('nc,bcd->bnd', self.points, hmat)
            transformed_points = transformed_points[:, :, :2] / transformed_points[:, :, 2:]
        
            floor_rx = tf.floor(transformed_points[:, :, 0:1])
            ceil_rx = tf.ceil(transformed_points[:, :, 0:1])
            floor_ry = tf.floor(transformed_points[:, :, 1:2])
            ceil_ry = tf.ceil(transformed_points[:, :, 1:2])

            r_tl = tf.cast(tf.concat((floor_rx, floor_ry), axis=2), tf.int32)
            r_tr = tf.cast(tf.concat((ceil_rx, floor_ry), axis=2), tf.int32)
            r_bl = tf.cast(tf.concat((floor_rx, ceil_ry), axis=2), tf.int32)
            r_br = tf.cast(tf.concat((ceil_rx, ceil_ry), axis=2), tf.int32)

            frac_x = transformed_points[:, :, 0:1] - floor_rx[:, :, 0:]
            frac_y = transformed_points[:, :, 1:2] - floor_ry[:, :, 0:]

            val_tl = get_values(im, r_tl) * (1 - frac_x) * (1 - frac_y)
            val_tr = get_values(im, r_tr) * frac_x * (1 - frac_y)
            val_bl = get_values(im, r_bl) * (1 - frac_x) * frac_y
            val_br = get_values(im, r_br) * frac_x * frac_y

            r = tf.reshape(val_tl + val_tr + val_bl + val_br, (-1, self.flags.height, self.flags.width, self.flags.channels))
        return r

class Encoder(object):
    def __init__(self, flags):
        self.flags = flags
        self.tensor_dict = {}

    def encoder_early_fusion(self, i1, i2, scope='enc_early'):
        with tf.variable_scope(scope):
            W_ec1 = tf.get_variable('W_ec1', (9, 9, 2*self.flags.channels, 32), initializer=tf.contrib.layers.xavier_initializer())
            b_ec1 = tf.get_variable('b_ec1', (32), initializer=tf.constant_initializer(0.01))
            W_ec2 = tf.get_variable('W_ec2', (7, 7, 32, 64), initializer=tf.contrib.layers.xavier_initializer())
            b_ec2 = tf.get_variable('b_ec2', (64), initializer=tf.constant_initializer(0.01))
            W_ec3 = tf.get_variable('W_ec3', (5, 5, 64, 128), initializer=tf.contrib.layers.xavier_initializer())
            b_ec3 = tf.get_variable('b_ec3', (128), initializer=tf.constant_initializer(0.01))
            W_ec4 = tf.get_variable('W_ec4', (3, 3, 128, 256), initializer=tf.contrib.layers.xavier_initializer())
            b_ec4 = tf.get_variable('b_ec4', (256), initializer=tf.constant_initializer(0.01))
            W_ec5 = tf.get_variable('W_ec5', (3, 3, 256, 512), initializer=tf.contrib.layers.xavier_initializer())
            b_ec5 = tf.get_variable('b_ec5', (512), initializer=tf.constant_initializer(0.01))
            W_ec6 = tf.get_variable('W_ec6', (1, 1, 512, 1024), initializer=tf.contrib.layers.xavier_initializer())
            b_ec6 = tf.get_variable('b_ec6', (1024), initializer=tf.constant_initializer(0.01))

            Si = tf.concat((i1, i2), 3)
            EC1 = tf.nn.relu(tf.nn.conv2d(Si, W_ec1, strides=[1, 1, 1, 1], padding='SAME') + b_ec1)
            EP1 = tf.nn.max_pool(EC1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
            EC2 = tf.nn.relu(tf.nn.conv2d(EP1, W_ec2, strides=[1, 1, 1, 1], padding='SAME') + b_ec2)
            EP2 = tf.nn.max_pool(EC2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
            EC3 = tf.nn.relu(tf.nn.conv2d(EP2, W_ec3, strides=[1, 1, 1, 1], padding='SAME') + b_ec3)
            EP3 = tf.nn.max_pool(EC3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
            EC4 = tf.nn.relu(tf.nn.conv2d(EP3, W_ec4, strides=[1, 1, 1, 1], padding='SAME') + b_ec4)
            EP4 = tf.nn.max_pool(EC4, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
            EC5 = tf.nn.relu(tf.nn.conv2d(EP4, W_ec5, strides=[1, 1, 1, 1], padding='SAME') + b_ec5)
            EP5 = tf.nn.max_pool(EC5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
            EC6 = tf.nn.relu(tf.nn.conv2d(EP5, W_ec6, strides=[1, 1, 1, 1], padding='SAME') + b_ec6)
            
            return EC3, EC4, EC5, EC6

    def encoder_late_fusion(self, i1, i2, scope='enc_late'):
        def conv_tower(im):
            W_ec1 = tf.get_variable('W_ec1', (9, 9, self.flags.channels, 32), initializer=tf.contrib.layers.xavier_initializer())
            b_ec1 = tf.get_variable('b_ec1', (32), initializer=tf.constant_initializer(0.01))
            W_ec2 = tf.get_variable('W_ec2', (7, 7, 32, 64), initializer=tf.contrib.layers.xavier_initializer())
            b_ec2 = tf.get_variable('b_ec2', (64), initializer=tf.constant_initializer(0.01))
            W_ec3 = tf.get_variable('W_ec3', (5, 5, 64, 128), initializer=tf.contrib.layers.xavier_initializer())
            b_ec3 = tf.get_variable('b_ec3', (128), initializer=tf.constant_initializer(0.01))
            W_ec4 = tf.get_variable('W_ec4', (3, 3, 128, 256), initializer=tf.contrib.layers.xavier_initializer())
            b_ec4 = tf.get_variable('b_ec4', (256), initializer=tf.constant_initializer(0.01))
            W_ec5 = tf.get_variable('W_ec5', (3, 3, 256, 512), initializer=tf.contrib.layers.xavier_initializer())
            b_ec5 = tf.get_variable('b_ec5', (512), initializer=tf.constant_initializer(0.01))
            W_ec6 = tf.get_variable('W_ec6', (1, 1, 512, 512), initializer=tf.contrib.layers.xavier_initializer())
            b_ec6 = tf.get_variable('b_ec6', (512), initializer=tf.constant_initializer(0.01))

            EC1 = tf.nn.relu(tf.nn.conv2d(im, W_ec1, strides=[1, 1, 1, 1], padding='SAME') + b_ec1)
            EP1 = tf.nn.max_pool(EC1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
            EC2 = tf.nn.relu(tf.nn.conv2d(EP1, W_ec2, strides=[1, 1, 1, 1], padding='SAME') + b_ec2)
            EP2 = tf.nn.max_pool(EC2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
            EC3 = tf.nn.relu(tf.nn.conv2d(EP2, W_ec3, strides=[1, 1, 1, 1], padding='SAME') + b_ec3)
            EP3 = tf.nn.max_pool(EC3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
            EC4 = tf.nn.relu(tf.nn.conv2d(EP3, W_ec4, strides=[1, 1, 1, 1], padding='SAME') + b_ec4)
            EP4 = tf.nn.max_pool(EC4, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
            EC5 = tf.nn.relu(tf.nn.conv2d(EP4, W_ec5, strides=[1, 1, 1, 1], padding='SAME') + b_ec5)
            EP5 = tf.nn.max_pool(EC5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
            EC6 = tf.nn.relu(tf.nn.conv2d(EP5, W_ec6, strides=[1, 1, 1, 1], padding='SAME') + b_ec6)
            
            return EC3, EC4, EC5, EC6
            
        with tf.variable_scope(scope):
            EC3_1, EC4_1, EC5_1, EC6_1 = conv_tower(i1)
            tf.get_variable_scope().reuse_variables()
            EC3_2, EC4_2, EC5_2, EC6_2 = conv_tower(i2)

            EC3_1_2 = tf.concat((EC3_1, EC3_2), 3)
            EC4_1_2 = tf.concat((EC4_1, EC4_2), 3)
            EC5_1_2 = tf.concat((EC5_1, EC5_2), 3)
            EC6_1_2 = tf.concat((EC6_1, EC6_2), 3)

        return EC3_1_2, EC4_1_2, EC5_1_2, EC6_1_2

class Decoder(object):
    def __init__(self, flags):
        self.flags = flags
        self.tensor_dict = {}      

    def correspondence_decoder(self, EC6, EC5, EC4, EC3, scope='correspondence_decoder'):
        with tf.variable_scope(scope):
            W_ec3_feature = tf.get_variable('W_ec3_feature', (1, 1, EC3.shape[3], 64), initializer=tf.contrib.layers.xavier_initializer())
            b_ec3_feature = tf.get_variable('b_ec3_feature', (64), initializer=tf.constant_initializer(0.01))
            W_ec4_feature = tf.get_variable('W_ec4_feature', (1, 1, EC4.shape[3], 128), initializer=tf.contrib.layers.xavier_initializer())
            b_ec4_feature = tf.get_variable('b_ec4_feature', (128), initializer=tf.constant_initializer(0.01))
            W_ec5_feature = tf.get_variable('W_ec5_feature', (1, 1, EC5.shape[3], 256), initializer=tf.contrib.layers.xavier_initializer())
            b_ec5_feature = tf.get_variable('b_ec5_feature', (256), initializer=tf.constant_initializer(0.01))

            EC3_feature = tf.nn.relu(tf.nn.conv2d(EC3, W_ec3_feature, strides=[1, 1, 1, 1], padding='SAME') + b_ec3_feature)
            EC4_feature = tf.nn.relu(tf.nn.conv2d(EC4, W_ec4_feature, strides=[1, 1, 1, 1], padding='SAME') + b_ec4_feature)
            EC5_feature = tf.nn.relu(tf.nn.conv2d(EC5, W_ec5_feature, strides=[1, 1, 1, 1], padding='SAME') + b_ec5_feature)

            W_cc1 = tf.get_variable('W_cc1', (1, 1, 1024, 2048), initializer=tf.contrib.layers.xavier_initializer())
            b_cc1 = tf.get_variable('b_cc1', (2048), initializer=tf.constant_initializer(0.01))
            W_cc2 = tf.get_variable('W_cc2', (1, 1, 2048, 2048), initializer=tf.contrib.layers.xavier_initializer())
            b_cc2 = tf.get_variable('b_cc2', (2048), initializer=tf.constant_initializer(0.01))
            W_cd1 = tf.get_variable('W_cd1', (4, 4, 768, 2048), initializer=tf.contrib.layers.xavier_initializer())
            b_cd1 = tf.get_variable('b_cd1', (768), initializer=tf.constant_initializer(0.01))
            W_cd2 = tf.get_variable('W_cd2', (4, 4, 384, 1024), initializer=tf.contrib.layers.xavier_initializer())
            b_cd2 = tf.get_variable('b_cd2', (384), initializer=tf.constant_initializer(0.01))
            W_cd3 = tf.get_variable('W_cd3', (4, 4, 192, 512), initializer=tf.contrib.layers.xavier_initializer())
            b_cd3 = tf.get_variable('b_cd3', (192), initializer=tf.constant_initializer(0.01))
            W_cd4 = tf.get_variable('W_cd4', (4, 4, 128, 256), initializer=tf.contrib.layers.xavier_initializer())
            b_cd4 = tf.get_variable('b_cd4', (128), initializer=tf.constant_initializer(0.01))
            W_cd5 = tf.get_variable('W_cd5', (4, 4, 64, 128), initializer=tf.contrib.layers.xavier_initializer())
            b_cd5 = tf.get_variable('b_cd5', (64), initializer=tf.constant_initializer(0.01))
            W_cc3 = tf.get_variable('W_cc3', (3, 3, 64, 1), initializer=tf.contrib.layers.xavier_initializer())
            b_cc3 = tf.get_variable('b_cc3', (1), initializer=tf.constant_initializer(0.01))

            batch = tf.shape(EC6)[0]
            CC1 = tf.nn.relu(tf.nn.conv2d(EC6, W_cc1, strides=[1, 1, 1, 1], padding='SAME') + b_cc1)
            CC2 = tf.nn.relu(tf.nn.conv2d(CC1, W_cc2, strides=[1, 1, 1, 1], padding='SAME') + b_cc2)
            CD1 = tf.nn.relu(tf.nn.conv2d_transpose(CC2, W_cd1, [batch, 16, 16, 768], strides=[1, 2, 2, 1], padding='SAME') + b_cd1)
            CD1_EC5 = tf.concat((CD1, EC5_feature), 3)
            CD2 = tf.nn.relu(tf.nn.conv2d_transpose(CD1_EC5, W_cd2, [batch, 32, 32, 384], strides=[1, 2, 2, 1], padding='SAME') + b_cd2)
            CD2_EC4 = tf.concat((CD2, EC4_feature), 3)
            CD3 = tf.nn.relu(tf.nn.conv2d_transpose(CD2_EC4, W_cd3, [batch, 64, 64, 192], strides=[1, 2, 2, 1], padding='SAME') + b_cd3)
            CD3_EC3 = tf.concat((CD3, EC3_feature), 3)
            CD4 = tf.nn.relu(tf.nn.conv2d_transpose(CD3_EC3, W_cd4, [batch, 128, 128, 128], strides=[1, 2, 2, 1], padding='SAME') + b_cd4)
            CD5 = tf.nn.relu(tf.nn.conv2d_transpose(CD4, W_cd5, [batch, 256, 256, 64], strides=[1, 2, 2, 1], padding='SAME') + b_cd5)
            CC3 = tf.nn.conv2d(CD5, W_cc3, strides=[1, 1, 1, 1], padding='SAME') + b_cc3

        return CC3

    def visibility_decoder(self, EC6, scope='visibility_decoder'):
        with tf.variable_scope(scope):           
            W_vc1 = tf.get_variable('W_vc1', (1, 1, 1024, 1024), initializer=tf.contrib.layers.xavier_initializer())
            b_vc1 = tf.get_variable('b_vc1', (1024), initializer=tf.constant_initializer(0.01))
            W_vc2 = tf.get_variable('W_vc2', (1, 1, 1024, 1024), initializer=tf.contrib.layers.xavier_initializer())
            b_vc2 = tf.get_variable('b_vc2', (1024), initializer=tf.constant_initializer(0.01))
            W_vd1 = tf.get_variable('W_vd1', (4, 4, 512, 1024), initializer=tf.contrib.layers.xavier_initializer())
            b_vd1 = tf.get_variable('b_vd1', (512), initializer=tf.constant_initializer(0.01))
            W_vd2 = tf.get_variable('W_vd2', (4, 4, 256, 512), initializer=tf.contrib.layers.xavier_initializer())
            b_vd2 = tf.get_variable('b_vd2', (256), initializer=tf.constant_initializer(0.01))
            W_vd3 = tf.get_variable('W_vd3', (4, 4, 128, 256), initializer=tf.contrib.layers.xavier_initializer())
            b_vd3 = tf.get_variable('b_vd3', (128), initializer=tf.constant_initializer(0.01))
            W_vd4 = tf.get_variable('W_vd4', (4, 4, 64, 128), initializer=tf.contrib.layers.xavier_initializer())
            b_vd4 = tf.get_variable('b_vd4', (64), initializer=tf.constant_initializer(0.01))
            W_vd5 = tf.get_variable('W_vd5', (4, 4, 32, 64), initializer=tf.contrib.layers.xavier_initializer())
            b_vd5 = tf.get_variable('b_vd5', (32), initializer=tf.constant_initializer(0.01))
            W_vc3 = tf.get_variable('W_vc3', (3, 3, 32, 1), initializer=tf.contrib.layers.xavier_initializer())
            b_vc3 = tf.get_variable('b_vc3', (1), initializer=tf.constant_initializer(0.01))
        
            batch = tf.shape(EC6)[0]
            VC1 = tf.nn.relu(tf.nn.conv2d(EC6, W_vc1, strides=[1, 1, 1, 1], padding='SAME') + b_vc1)
            VC2 = tf.nn.relu(tf.nn.conv2d(VC1, W_vc2, strides=[1, 1, 1, 1], padding='SAME') + b_vc2)
            VD1 = tf.nn.relu(tf.nn.conv2d_transpose(VC2, W_vd1, [batch, 16, 16, 512], strides=[1, 2, 2, 1], padding='SAME') + b_vd1)
            VD2 = tf.nn.relu(tf.nn.conv2d_transpose(VD1, W_vd2, [batch, 32, 32, 256], strides=[1, 2, 2, 1], padding='SAME') + b_vd2)
            VD3 = tf.nn.relu(tf.nn.conv2d_transpose(VD2, W_vd3, [batch, 64, 64, 128], strides=[1, 2, 2, 1], padding='SAME') + b_vd3)
            VD4 = tf.nn.relu(tf.nn.conv2d_transpose(VD3, W_vd4, [batch, 128, 128, 64], strides=[1, 2, 2, 1], padding='SAME') + b_vd4)
            VD5 = tf.nn.relu(tf.nn.conv2d_transpose(VD4, W_vd5, [batch, 256, 256, 32], strides=[1, 2, 2, 1], padding='SAME') + b_vd5)
            VC3 = tf.nn.conv2d(VD5, W_vc3, strides=[1, 1, 1, 1], padding='SAME') + b_vc3
            VC3_sig = tf.nn.sigmoid(VC3)

        return VC3_sig

    def depthmap_decoder(self, EC6, EC5, EC4, EC3, scope='depthmap_decoder'):
        with tf.variable_scope(scope):
            W_ec3_feature = tf.get_variable('W_ec3_feature', (1, 1, EC3.shape[3], 64), initializer=tf.contrib.layers.xavier_initializer())
            b_ec3_feature = tf.get_variable('b_ec3_feature', (64), initializer=tf.constant_initializer(0.01))
            W_ec4_feature = tf.get_variable('W_ec4_feature', (1, 1, EC4.shape[3], 128), initializer=tf.contrib.layers.xavier_initializer())
            b_ec4_feature = tf.get_variable('b_ec4_feature', (128), initializer=tf.constant_initializer(0.01))
            W_ec5_feature = tf.get_variable('W_ec5_feature', (1, 1, EC5.shape[3], 256), initializer=tf.contrib.layers.xavier_initializer())
            b_ec5_feature = tf.get_variable('b_ec5_feature', (256), initializer=tf.constant_initializer(0.01))

            EC3_feature = tf.nn.relu(tf.nn.conv2d(EC3, W_ec3_feature, strides=[1, 1, 1, 1], padding='SAME') + b_ec3_feature)
            EC4_feature = tf.nn.relu(tf.nn.conv2d(EC4, W_ec4_feature, strides=[1, 1, 1, 1], padding='SAME') + b_ec4_feature)
            EC5_feature = tf.nn.relu(tf.nn.conv2d(EC5, W_ec5_feature, strides=[1, 1, 1, 1], padding='SAME') + b_ec5_feature)

            W_dc1 = tf.get_variable('W_dc1', (1, 1, 1024, 2048), initializer=tf.contrib.layers.xavier_initializer())
            b_dc1 = tf.get_variable('b_dc1', (2048), initializer=tf.constant_initializer(0.01))
            W_dc2 = tf.get_variable('W_dc2', (1, 1, 2048, 2048), initializer=tf.contrib.layers.xavier_initializer())
            b_dc2 = tf.get_variable('b_dc2', (2048), initializer=tf.constant_initializer(0.01))
            W_dd1 = tf.get_variable('W_dd1', (4, 4, 768, 2048), initializer=tf.contrib.layers.xavier_initializer())
            b_dd1 = tf.get_variable('b_dd1', (768), initializer=tf.constant_initializer(0.01))
            W_dd2 = tf.get_variable('W_dd2', (4, 4, 384, 1024), initializer=tf.contrib.layers.xavier_initializer())
            b_dd2 = tf.get_variable('b_dd2', (384), initializer=tf.constant_initializer(0.01))
            W_dd3 = tf.get_variable('W_dd3', (4, 4, 192, 512), initializer=tf.contrib.layers.xavier_initializer())
            b_dd3 = tf.get_variable('b_dd3', (192), initializer=tf.constant_initializer(0.01))
            W_dd4 = tf.get_variable('W_dd4', (4, 4, 128, 256), initializer=tf.contrib.layers.xavier_initializer())
            b_dd4 = tf.get_variable('b_dd4', (128), initializer=tf.constant_initializer(0.01))
            W_dd5 = tf.get_variable('W_dd5', (4, 4, 64, 128), initializer=tf.contrib.layers.xavier_initializer())
            b_dd5 = tf.get_variable('b_dd5', (64), initializer=tf.constant_initializer(0.01))
            W_dc3 = tf.get_variable('W_dc3', (3, 3, 64, 1), initializer=tf.contrib.layers.xavier_initializer())
            b_dc3 = tf.get_variable('b_dc3', (1), initializer=tf.constant_initializer(0.01))

            batch = tf.shape(EC6)[0]
            DC1 = tf.nn.relu(tf.nn.conv2d(EC6, W_dc1, strides=[1, 1, 1, 1], padding='SAME') + b_dc1)
            DC2 = tf.nn.relu(tf.nn.conv2d(DC1, W_dc2, strides=[1, 1, 1, 1], padding='SAME') + b_dc2)
            DD1 = tf.nn.relu(tf.nn.conv2d_transpose(DC2, W_dd1, [batch, 16, 16, 768], strides=[1, 2, 2, 1], padding='SAME') + b_dd1)
            DD1_EC5 = tf.concat((DD1, EC5_feature), 3)
            DD2 = tf.nn.relu(tf.nn.conv2d_transpose(DD1_EC5, W_dd2, [batch, 32, 32, 384], strides=[1, 2, 2, 1], padding='SAME') + b_dd2)
            DD2_EC4 = tf.concat((DD2, EC4_feature), 3)
            DD3 = tf.nn.relu(tf.nn.conv2d_transpose(DD2_EC4, W_dd3, [batch, 64, 64, 192], strides=[1, 2, 2, 1], padding='SAME') + b_dd3)
            DD3_EC3 = tf.concat((DD3, EC3_feature), 3)
            DD4 = tf.nn.relu(tf.nn.conv2d_transpose(DD3_EC3, W_dd4, [batch, 128, 128, 128], strides=[1, 2, 2, 1], padding='SAME') + b_dd4)
            DD5 = tf.nn.relu(tf.nn.conv2d_transpose(DD4, W_dd5, [batch, 256, 256, 64], strides=[1, 2, 2, 1], padding='SAME') + b_dd5)
            DC3 = tf.nn.conv2d(DD5, W_dc3, strides=[1, 1, 1, 1], padding='SAME') + b_dc3
            DC3_relu = tf.nn.relu(DC3)

        return DC3_relu

class ViewMorphing(object):
    def __init__(self, flags):
        self.flags = flags
        self.tensor_dict = {}

        height, width = (self.flags.height, self.flags.width)
        points = np.reshape(np.arange(height * width), (height, width, 1))
        self.points = tf.constant(points, dtype=tf.float32)

    def morph(self, r1, r2, c, m, scope='morph'):
        def get_values(im, p):
            shape = tf.shape(im)
            pixels = self.flags.height * self.flags.width
            im = tf.reshape(im, (-1, pixels, self.flags.channels))
            p = tf.expand_dims(tf.reshape(p, (-1, pixels)), axis=2)
           
            idx = tf.expand_dims(tf.range(start=0, limit=shape[0], delta=1), axis=1)
            idx = tf.expand_dims(tf.tile(idx, [1, pixels]), axis=2)
            idx = tf.concat((idx, tf.cast(p, tf.int32)), axis=2)
            
            values = tf.gather_nd(im, idx)
            return tf.reshape(values, shape)
            
        with tf.variable_scope(scope):
            p1 = self.points + c
            p2 = self.points - c

            floor_p1 = tf.floor(p1)
            ceil_p1 = tf.ceil(p1)
            floor_p2 = tf.floor(p2)
            ceil_p2 = tf.ceil(p2)
        
            frac_p1 = p1 - floor_p1
            frac_p2 = p2 - floor_p2

            r1_p1 = get_values(r1, floor_p1) * (1 - frac_p1) + get_values(r1, ceil_p1) * frac_p1
            r2_p2 = get_values(r2, floor_p2) * (1 - frac_p2) + get_values(r2, ceil_p2) * frac_p2
     
            r = r1_p1 * m + r2_p2 * (1 - m)

        self.tensor_dict['R1(P1)'] = r1_p1
        self.tensor_dict['R2(P2)'] = r2_p2
        return r

class DVMSystem(BaseDVMSystem):
    def __init__(self, flags):
        super(DVMSystem, self).__init__(flags)
        self.model_name = 'dvm_model'

    def setup_system(self):
        self.tensor_dict = {}
        
        self.r = Rectification(self.flags)
        self.e = Encoder(self.flags)
        self.d = Decoder(self.flags)
        self.vm = ViewMorphing(self.flags)

        h1, h2 = self.r.rectification_early_fusion(self.i1_placeholder, self.i2_placeholder)
        r1 = self.r.rectification(self.i1_placeholder, h1)
        r2 = self.r.rectification(self.i2_placeholder, h2)

        ec3, ec4, ec5, ec6 = self.e.encoder_late_fusion(r1, r2)

        c = self.d.correspondence_decoder(ec6, ec5, ec4, ec3)
        m = self.d.visibility_decoder(ec6)
        if self.flags.use_depth:
            dm = self.d.depthmap_decoder(ec6, ec5, ec4, ec3)
     
        self.R = self.vm.morph(r1, r2, c, m)
        if self.flags.use_depth:
            self.DM = tf.squeeze(dm, axis=3)

        self.tensor_dict['H1'] = h1
        self.tensor_dict['H2'] = h2
        self.tensor_dict['R1'] = r1
        self.tensor_dict['R2'] = r2
        self.tensor_dict['EC3'] = ec3
        self.tensor_dict['EC4'] = ec4
        self.tensor_dict['EC5'] = ec5
        self.tensor_dict['EC6'] = ec6
        self.tensor_dict['C'] = c
        self.tensor_dict['M'] = m

