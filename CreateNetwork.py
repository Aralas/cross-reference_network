# -*- coding:utf-8 -*-
"""
@author:Xu Jingyi
@file:CreateNetwork.py
@time:2018/11/2722:28
"""

import tensorflow as tf


class CreateNetwork(object):

    def __init__(self, architecture, input_shape, dropout, num_classes):
        self.architecture = architecture
        self.input_shape = input_shape
        self.dropout = dropout
        self.num_classes = num_classes


class CreateCNN(CreateNetwork):

    def __init__(self, architecture, input_shape, dropout, num_classes):
        CreateNetwork.__init__(architecture, input_shape, dropout, num_classes)
        self.model = self.generate_model()

    def generate_model(self):
        model = tf.contrib.keras.models.Sequential()
        for layer_index in range(length(self.architecture)):
            layer = self.architecture[layer_index]
            if length(layer) == 3:
                model.add(tf.contrib.keras.layers.Conv2D(layer[0], kernel_size=(layer[1], layer[2])),
                          kernel_initializer=tf.keras.initializers.glorot_normal, activation='relu',
                          input_shape=self.input_shape)
                model.add(tf.contrib.keras.layers.MaxPooling2D(pool_size=(2, 2)))
            elif length(layer) == 1:
                if length(self.architecture[layer_index - 1]) == 3:
                    model.add(tf.contrib.keras.layers.Flatten())
                model.add(tf.contrib.keras.layers.Dense(layer[0], activation='relu',
                                                        kernel_initializer=tf.keras.initializers.glorot_normal))
            else:
                print('Invalid architecture /(ㄒoㄒ)/~~')
        model.add(tf.contrib.keras.layers.Dropout(self.dropout))
        if self.num_classes > 2:
            activation_for_output = 'softmax'
        elif self.num_classes == 2:
            activation_for_output = 'sigmoid'
        model.add(tf.contrib.keras.layers.Dense(self.num_classes, activation=activation_for_output))
        model.compile(loss=tf.contrib.keras.losses.categorical_crossentropy, metrics=['accuracy'],
                      optimizer=tf.contrib.keras.optimizers.Adam)
        return model

    def train_model(self, x, y, batch_size, epochs):
        self.model.fit(x, y, batch_size=batch_size, epochs=epochs)
        return score


class CreateFullyConnected(CreateNetwork):

    def __init__(self, architecture, input_shape, dropout, num_classes):
        CreateNetwork.__init__(architecture, input_shape, dropout, num_classes)
        self.model = self.generate_model()

    def generate_model(self):
        model = tf.contrib.keras.models.Sequential()
        for layer in self.architecture:
            if length(layer[0]) == 1:
                model.add(tf.contrib.keras.layers.Dense(layer[0], activation='relu',
                                                        kernel_initializer=tf.keras.initializers.glorot_normal))
            else:
                print('Invalid architecture /(ㄒoㄒ)/~~')
        model.add(tf.contrib.keras.layers.Dropout(self.dropout))
        if self.num_classes > 2:
            activation_for_output = 'softmax'
        elif self.num_classes == 2:
            activation_for_output = 'sigmoid'
        model.add(tf.contrib.keras.layers.Dense(self.num_classes, activation=activation_for_output))
        model.compile(loss=tf.contrib.keras.losses.categorical_crossentropy, metrics=['accuracy'],
                      optimizer=tf.contrib.keras.optimizers.Adam)
        return model

    def train_model(self, x, y, batch_size, epochs):
        self.model.fit(x, y, batch_size=batch_size, epochs=epochs)
        return score
