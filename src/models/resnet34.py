import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import \
    GlobalAveragePooling2D, Add, \
    Dense, Activation, Flatten, \
    BatchNormalization, Conv2D, MaxPooling2D
from sklearn.utils import shuffle

# tf.log(0)によるnanを防ぐ
def tf_log(x):
    return tf.log(tf.clip_by_value(x, 1e-10, x))

class ResNet34(Model):
    '''
    Reference:
        "Deep Residual Learning for Image Recognition"
        https://arxiv.org/abs/1512.03385
    '''
    def __init__(self, input_shape, output_dim):
        super().__init__()
        self.conv1 = Conv2D(64, input_shape=input_shape,
                            kernel_size=(7, 7),
                            strides=(2, 2),
                            padding='same')
        self.bn1 = BatchNormalization()
        self.relu1 = Activation('relu')
        self.pool1 = MaxPooling2D(pool_size=(3, 3),
                                  strides=(2, 2),
                                  padding='same')
        self.block1 = [
            self._building_block(64) for _ in range(3)
        ]
        self.conv2 = Conv2D(128,
                            kernel_size=(1, 1),
                            strides=(2, 2))
        self.block2 = [
            self._building_block(128) for _ in range(4)
        ]
        self.conv3 = Conv2D(256,
                            kernel_size=(1, 1),
                            strides=(2, 2))
        self.block3 = [
            self._building_block(256) for _ in range(6)
        ]
        self.conv4 = Conv2D(512,
                            kernel_size=(1, 1),
                            strides=(2, 2))
        self.block4 = [
            self._building_block(512) for _ in range(3)
        ]
        self.avg_pool = GlobalAveragePooling2D()
        self.fc = Dense(1000, activation='relu')
        self.out = Dense(output_dim, activation='softmax')

    def build(self):
        x = tf.placeholder(shape=[None, 32, 32, 3], dtype=tf.float32)
        t = tf.placeholder(tf.float32, [None, 10])
        is_training = tf.placeholder(tf.bool, shape=()) # batch normalizationのために定義
        
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu1(h)
        h = self.pool1(h)
        for block in self.block1:
            h = block(h)
        h = self.conv2(h)
        for block in self.block2:
            h = block(h)
        h = self.conv3(h)
        for block in self.block3:
            h = block(h)
        h = self.conv4(h)
        for block in self.block4:
            h = block(h)
        h = self.avg_pool(h)
        h = self.fc(h)
        y = self.out(h)

        loss = - tf.reduce_mean(tf.reduce_sum(t * tf_log(y), axis=1))

        correct = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1)) # 学習部分で正解率の計算はするか？
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        return x, t, is_training, y, loss, accuracy
        
    def _building_block(self, channel_out=64):
        return Block(channel_out=channel_out)


class Block(Model):
    def __init__(self, channel_out=64):
        super().__init__()
        self.conv1 = Conv2D(channel_out,
                            kernel_size=(3, 3),
                            padding='same')
        self.bn1 = BatchNormalization()
        self.relu1 = Activation('relu')
        self.conv2 = Conv2D(channel_out,
                            kernel_size=(3, 3),
                            padding='same')
        self.bn2 = BatchNormalization()
        self.add = Add()
        self.relu2 = Activation('relu')

    def call(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu1(h)
        h = self.conv2(h)
        h = self.bn2(h)
        h = self.add([x, h])
        y = self.relu2(h)
        return y