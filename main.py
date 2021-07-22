import tensorflow as tf
import tensorflow.contrib as tf_contrib


##################################################################################
# Layer
##################################################################################

def conv(x, channels, kernel=4, stride=2, padding='SAME', use_bias=True, scope='conv_0'):
    with tf.variable_scope(scope):
        x = tf.layers.conv2d(inputs=x, filters=channels,
                             kernel_size=kernel, kernel_initializer=weight_init,
                             kernel_regularizer=weight_regularizer,
                             strides=stride, use_bias=use_bias, padding=padding)

        return x


def fully_conneted(x, units, use_bias=True, scope='fully_0'):
    with tf.variable_scope(scope):
        x = flatten(x)
        x = tf.layers.dense(x, units=units, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer,
                            use_bias=use_bias)

        return x


def resblock(x_init, channel_cate, spatial_cate, channels, is_training=True, use_bias=True, downsample=False,
             scope='resblock'):
    with tf.variable_scope(scope):

        x = batch_norm(x_init, is_training, scope='batch_norm_0')
        x = relu(x)

        if downsample:
            x = conv(x, channels, kernel=3, stride=2, use_bias=use_bias, scope='conv_0')
            x_init = conv(x_init, channels, kernel=1, stride=2, use_bias=use_bias, scope='conv_init')

        else:
            x = conv(x, channels, kernel=3, stride=1, use_bias=use_bias, scope='conv_0')

        x = batch_norm(x, is_training, scope='batch_norm_1')
        x = relu(x)
        x = conv(x, channels, kernel=3, stride=1, use_bias=use_bias, scope='conv_1')
        x = cbam_block(x, 'cbam_CSCNN', channel_cate, spatial_cate, ratio=4)

        return x + x_init


def bottle_resblock(x_init, channels, is_training=True, use_bias=True, downsample=False, scope='bottle_resblock'):
    with tf.variable_scope(scope):
        x = batch_norm(x_init, is_training, scope='batch_norm_1x1_front')
        shortcut = relu(x)

        x = conv(shortcut, channels, kernel=1, stride=1, use_bias=use_bias, scope='conv_1x1_front')
        x = batch_norm(x, is_training, scope='batch_norm_3x3')
        x = relu(x)

        if downsample:
            x = conv(x, channels, kernel=3, stride=2, use_bias=use_bias, scope='conv_0')
            shortcut = conv(shortcut, channels * 4, kernel=1, stride=2, use_bias=use_bias, scope='conv_init')

        else:
            x = conv(x, channels, kernel=3, stride=1, use_bias=use_bias, scope='conv_0')
            shortcut = conv(shortcut, channels * 4, kernel=1, stride=1, use_bias=use_bias, scope='conv_init')

        x = batch_norm(x, is_training, scope='batch_norm_1x1_back')
        x = relu(x)
        x = conv(x, channels * 4, kernel=1, stride=1, use_bias=use_bias, scope='conv_1x1_back')

        return x + shortcut


def get_residual_layer(res_n):
    x = []

    if res_n == 18:
        x = [2, 2, 2, 2]

    if res_n == 34:
        x = [3, 4, 6, 3]

    if res_n == 50:
        x = [3, 4, 6, 3]

    if res_n == 101:
        x = [3, 4, 23, 3]

    if res_n == 152:
        x = [3, 8, 36, 3]

    return x


def flatten(x):
    return tf.layers.flatten(x)


def global_avg_pooling(x):
    gap = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    return gap


def avg_pooling(x):
    return tf.layers.average_pooling2d(x, pool_size=2, strides=2, padding='SAME')


def relu(x):
    return tf.nn.relu(x)


def batch_norm(x, is_training=True, scope='batch_norm'):
    return tf_contrib.layers.batch_norm(x,
                                        decay=0.9, epsilon=1e-05,
                                        center=True, scale=True, updates_collections=None,
                                        is_training=is_training, scope=scope)


class ResNet(object):
    def __init__(self):
        self.model_name = 'ResNet'
        self.res_n = 18
        self.embedding_dim = 60

    ##################################################################################
    # Generator
    ##################################################################################

    def network(self, x, att_cate, spatial_cate, is_training=True, reuse=False):
        with tf.variable_scope("network", reuse=reuse):
            if self.res_n < 50:
                residual_block = resblock
            else:
                residual_block = bottle_resblock

            residual_list = get_residual_layer(self.res_n)

            ch = 4  # paper is 64
            x = conv(x, channels=ch, kernel=3, stride=1, scope='conv')

            for i in range(residual_list[0]):
                x = residual_block(x, channels=ch, is_training=is_training, downsample=False,
                                   scope='resblock0_' + str(i))

            ########################################################################################################

            x = residual_block(x, att_cate, spatial_cate, channels=ch * 2, is_training=is_training, downsample=True,
                               scope='resblock1_0')

            for i in range(1, residual_list[1]):
                x = residual_block(x, channels=ch * 2, is_training=is_training, downsample=False,
                                   scope='resblock1_' + str(i))

            ########################################################################################################

            x = residual_block(x, att_cate, spatial_cate, channels=ch * 4, is_training=is_training, downsample=True,
                               scope='resblock2_0')

            for i in range(1, residual_list[2]):
                x = residual_block(x, channels=ch * 4, is_training=is_training, downsample=False,
                                   scope='resblock2_' + str(i))

            ########################################################################################################

            x = residual_block(x, att_cate, spatial_cate, channels=ch * 8, is_training=is_training, downsample=True,
                               scope='resblock_3_0')

            for i in range(1, residual_list[3]):
                x = residual_block(x, channels=ch * 8, is_training=is_training, downsample=False,
                                   scope='resblock_3_' + str(i))

            ########################################################################################################

            x = batch_norm(x, is_training, scope='batch_norm')
            x = relu(x)

            x = global_avg_pooling(x)
            x = fully_conneted(x, units=self.embedding_dim, scope='logit')

            return x


def CSCNN(img, channel_cate, spatial_cate):
    """
    CSCNN unofficial implementation
    """
    resnet = ResNet()
    img_CSCNN_embedding = resnet.network(img[:, :, :, :], channel_cate, spatial_cate)
    return img_CSCNN_embedding
