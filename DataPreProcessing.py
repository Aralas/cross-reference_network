# -*- coding:utf-8 -*-
"""
@author:Xu Jingyi
@file:DataPreProcessing.py
@time:2018/11/2722:29
"""
import tensorflow as tf


class LoadData(object):

    def __init__(self, seed, noise_level, augmentation):
        self.seed = seed
        self.noise_level = noise_level
        self.augmentation = augmentation

    def data_augmentation(self):
        pass

    def generate_noise_labels(self):
        pass


class MNIST(LoadData):

    def __init__(self, seed, noise_level, augmentation):
        LoadData.__init__(seed, noise_level, augmentation)
        self.num_classes = 10
        self.img_rows, self.img_cols = 28, 28
        self.input_size = [28, 28]

    def load_data(self):
        # load data
        mnist = tf.contrib.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        x_train = x_train.reshape(x_train.shape[0], self.img_rows, self.img_cols, 1)

        # transform labels to one-hot vectors
        y_train = tf.contrib.keras.utils.to_categorical(y_train, self.num_classes)
        y_test = tf.contrib.keras.utils.to_categorical(y_test, self.num_classes)

        if self.augmentation:
            pass
        if self.noise_level > 0:
            pass
        return x_train, y_train, x_test, y_test


class CIFAR10(LoadData):

    def __init__(self, seed, noise_level, augmentation):
        LoadData.__init__(seed, noise_level, augmentation)

    def load_data(self):
        cifar10 = tf.keras.datasets.cifar10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        if self.augmentation:
            pass
        if self.noise_level > 0:
            pass
        return x_train, y_train, x_test, y_test


class NORB(LoadData):

    def __init__(self, seed, noise_level, augmentation):
        LoadData.__init__(seed, noise_level, augmentation)

    def load_data(self):
        pass


class NewsGroup(LoadData):

    def __init__(self, seed, noise_level, augmentation):
        LoadData.__init__(seed, noise_level, augmentation)

    def load_data(self):
        pass
