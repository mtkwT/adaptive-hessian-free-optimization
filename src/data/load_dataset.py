import numpy as np
import keras
import tensorflow as tf
from keras.datasets import mnist, fashion_mnist, cifar10, cifar100
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

# データロード
class DataLoader():
    
    def __init__(self, random_state=42):
        self.random_state = random_state

    def load_mnist(self):
        (train_X, train_y), (test_X, test_y) = mnist.load_data()

        X = np.r_[train_X, test_X]
        y = np.r_[train_y, test_y]

        X = X.astype('float64') / 255
        X = X.reshape(X.shape[0], 28, 28, 1)
        y = y.astype('int64').flatten()
        y = np_utils.to_categorical(y, 10)

        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=10000, random_state=self.random_state)

        return train_X, test_X, train_y, test_y
        
    def load_fashion_mnist(self):
        (train_X, train_y), (test_X, test_y) = fashion_mnist.load_data()

        X = np.r_[train_X, test_X]
        y = np.r_[train_y, test_y]

        X = X.astype('float64') / 255
        X = X.reshape(X.shape[0], 28, 28, 1)
        y = y.astype('int64').flatten()
        y = np_utils.to_categorical(y, 10)

        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=10000, random_state=self.random_state)

        return train_X, test_X, train_y, test_y
    
    def load_cifar10(self):
        (train_X, train_y), (test_X, test_y) = cifar10.load_data()

        X = np.r_[train_X, test_X]
        y = np.r_[train_y, test_y]

        X = X.astype('float64') / 255
        y = y.astype('int64').flatten()
        y = np_utils.to_categorical(y, 10)

        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=10000, random_state=self.random_state)

        return train_X, test_X, train_y, test_y

    def load_cifar100(self):
        (train_X, train_y), (test_X, test_y) = cifar100.load_data()

        X = np.r_[train_X, test_X]
        y = np.r_[train_y, test_y]

        X = X.astype('float64') / 255
        y = y.astype('int64').flatten()
        y = np_utils.to_categorical(y, 100)

        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=10000, random_state=self.random_state)

        return train_X, test_X, train_y, test_y

if __name__ == "__main__":
    train_X, test_X, train_y, test_y = DataLoader().load_mnist()
    print(train_X.shape)
    print(test_X.shape)
    print(train_y.shape)
    print(test_y.shape)
    print()

    train_X, test_X, train_y, test_y = DataLoader().load_fashion_mnist()
    print(train_X.shape)
    print(test_X.shape)
    print(train_y.shape)
    print(test_y.shape)
    print()

    train_X, test_X, train_y, test_y = DataLoader().load_cifar10()
    print(train_X.shape)
    print(test_X.shape)
    print(train_y.shape)
    print(test_y.shape)
    print()

    train_X, test_X, train_y, test_y = DataLoader().load_cifar100()
    print(train_X.shape)
    print(test_X.shape)
    print(train_y.shape)
    print(test_y.shape)
    