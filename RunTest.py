# -*- coding:utf-8 -*-
"""
@author:Xu Jingyi
@file:RunTest.py
@time:2018/11/2722:28
"""

import tensorflow as tf
import FactoryClass

dataset = 'MNIST'
model_type = 'CNN'
# initialization = 'xavier'
model_architecture = [[32, 5, 5], [64, 5, 5], [500]]
dropout = 0.5
learning_rate = 0.001
batch_size = 128
section_num = 20
epochs = 100
data_size = 500


def randomly_sample_binary_data(x, y, data_size, label):
    return x_small, y_small


def generate_combined_label(y, label, binary_classifier_list):
    return y_hat


def run_cross_reference():
    data_object = FactoryClass.ChooseDataset(dataset, seed, noise_level, augmentation)
    x_train, y_train, x_test, y_test = data_object.x_train, data_object.y_train, data_object.x_test, data_object.y_test

    num_classes = data_object.num_classes
    input_size = data_object.input_size

    binary_classifier_list = []
    for label in range(num_classes):
        model_object = FactoryClass.ChooseNetworkCreator(model_type, model_architecture, input_size, dropout, 2)
        binary_classifier_list.append(model_object.generate_model())

    with tf.Session() as sess:
        for section in range(section_num):
            for label in range(num_classes):
                model = binary_classifier_list[label]
                x, y = randomly_sample_binary_data(x_train, y_train, data_size, label)
                lambda_ = 1
                y_hat = generate_combined_label(y, label, binary_classifier_list)

