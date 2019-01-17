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
from keras import backend as K
import numpy as np
import functools


class CreateNetwork(object):

    def __init__(self, architecture, input_shape, learning_rate, dropout, num_classes):
        self.architecture = architecture
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.num_classes = num_classes
        self.reference_output = None
        self.lamb_weight = 0
        self.power_n = 2

    def phi_threshold(self, x):
        if x >= 0.5:
            return np.power(x-0.5, self.power_n)
        else:
            return 0

    def create_reference(self, reference_output):
        ref1 = np.array([[np.sum([self.phi_threshold(value) for value in line])] for line in reference_output])
        media_matrix = np.array([[self.phi_threshold(1-value) for value in line] for line in self.reference_output])
        ref2 = functools.reduce(lambda x, y: x * y, media_matrix.T)
        ref2 = ref2.T
        return ref1, ref2

    def cross_ref_loss(self, y_true, y_pred):
        if self.reference_output is None:
            print('There is no reference matrix!!!')
            return K.mean(K.square(y_pred - y_true), axis=-1)
        else:
            lamb_weight = K.variable(self.lamb_weight)
            ref1, ref2 = self.create_reference(self.reference_output)
            ref1 = K.variable(ref1)
            ref2 = K.variable(ref2)
            loss = K.mean(K.square(y_pred - y_true), axis=-1) + \
                   lamb_weight * K.mean(np.multiply(y_true, ref1.reshape((len(ref1), 1))) * K.square(y_pred)) + \
                   lamb_weight * K.mean(np.multiply((1 - y_true), ref2.reshape((len(ref2), 1))) * K.square(1 - y_pred))
            return loss


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
            model.compile(loss=self.cross_ref_loss, metrics=['accuracy'], optimizer=adam)
        return model

    def train_model(self, x, y, batch_size, epochs):
        adam = Adam(lr=self.learning_rate)
        self.model.compile(loss=self.cross_ref_loss, metrics=['accuracy'], optimizer=adam)
        self.model.fit(x, y, batch_size=batch_size, epochs=epochs)

    def evaluate_model(self, x, y):
        adam = Adam(lr=self.learning_rate)
        self.model.compile(loss=self.cross_ref_loss, metrics=['accuracy'], optimizer=adam)
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

