# -*- coding:utf-8 -*-
"""
@author:Xu Jingyi
@file:CreateNetwork.py
@time:2018/11/2722:28
"""

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import Adam


class CreateNetwork(object):

    def __init__(self, architecture, input_shape, learning_rate, dropout, num_classes):
        self.architecture = architecture
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.num_classes = num_classes


class CreateCNN(CreateNetwork):

    def __init__(self, architecture, input_shape, learning_rate, dropout, num_classes):
        CreateNetwork.__init__(self, architecture, input_shape, learning_rate, dropout, num_classes)
        self.model = self.generate_model()

    def generate_model(self):
        model = Sequential()
        for layer_index in range(len(self.architecture)):
            layer = self.architecture[layer_index]
            if len(layer) == 3:
                if layer_index == 0:
                    model.add(Conv2D(layer[0], kernel_size=(layer[1], layer[2]), input_shape=self.input_shape,
                                     kernel_initializer='glorot_normal', activation='relu', padding='same'))
                else:
                    model.add(Conv2D(layer[0], kernel_size=(layer[1], layer[2]), kernel_initializer='glorot_normal',
                                     activation='relu', padding='same'))
                model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
            elif len(layer) == 1:
                if len(self.architecture[layer_index - 1]) == 3:
                    model.add(Flatten())
                model.add(Dense(layer[0], activation='relu', kernel_initializer='glorot_normal'))
            else:
                print('Invalid architecture /(ㄒoㄒ)/~~')
        model.add(Dropout(self.dropout))
        if self.num_classes > 2:
            model.add(Dense(self.num_classes))
            model.add(Activation('softmax'))
            adam = Adam(lr=self.learning_rate)
            model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)
        elif self.num_classes == 2:
            model.add(Dense(1))
            model.add(Activation('sigmoid'))
            adam = Adam(lr=self.learning_rate)
            model.compile(loss='mean_squared_error', metrics=['accuracy'], optimizer=adam)
        return model

    def train_model(self, x, y, batch_size, epochs):
        self.model.fit(x, y, batch_size=batch_size, epochs=epochs)

    def evaluate_model(self, x, y):
        loss, accuracy = self.model.evaluate(x, y)
        return loss, accuracy

    def prediction(self, x):
        return (self.model.predict(x))


class CreateFullyConnected(CreateNetwork):

    def __init__(self, architecture, input_shape, dropout, num_classes):
        CreateNetwork.__init__(self, architecture, input_shape, dropout, num_classes)
        self.model = self.generate_model()

    def generate_model(self):
        model = tf.contrib.keras.models.Sequential()
        for layer in self.architecture:
            if len(layer[0]) == 1:
                model.add(tf.contrib.keras.layers.Dense(layer[0], activation='relu',
                                                        kernel_initializer='glorot_normal'))
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
