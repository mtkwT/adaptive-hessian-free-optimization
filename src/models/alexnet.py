import numpy as np
import tensorflow as tf

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

# tf.log(0)によるnanを防ぐ
def tf_log(x):
    return tf.log(tf.clip_by_value(x, 1e-10, x))

def build_alexnet(input_shape=[None, 32, 32, 3], output_shape=[None, 10]):
    tf.reset_default_graph()

    x = tf.placeholder(shape=input_shape, dtype=tf.float32)
    t = tf.placeholder(shape=output_shape, dtype=tf.float32)

    is_training = tf.placeholder(tf.bool, shape=()) # batch normalizationのために定義

    h = tf.keras.layers.Conv2D(96, kernel_size=(11, 11), strides=(4, 4), padding='valid', activation='relu')(x)
    h = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(h)
    h = tf.keras.layers.BatchNormalization()(h)

    h = tf.keras.layers.Conv2D(256, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(h)
    h = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(h)
    h = tf.keras.layers.BatchNormalization()(h)

    h = tf.keras.layers.Conv2D(384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(h)
    h = tf.keras.layers.Conv2D(384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(h)
    h = tf.keras.layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(h)
    h = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(h)
    h = tf.keras.layers.BatchNormalization()(h)

    h = tf.keras.layers.Flatten()(h)
    h = tf.keras.layers.Dense(4096,activation = "relu")(h)
    h = tf.keras.layers.Dropout(rate=0.5)(h)
    h = tf.keras.layers.Dense(4096,activation = "relu")(h)
    h = tf.keras.layers.Dropout(rate=0.5)(h)
    y = tf.keras.layers.Dense(output_shape[1] ,activation = "softmax")(h)

    loss = - tf.reduce_mean(tf.reduce_sum(t * tf_log(y), axis=1))

    correct = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1)) # 学習部分で正解率の計算はするか？
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    return x, t, is_training, y, loss, accuracy