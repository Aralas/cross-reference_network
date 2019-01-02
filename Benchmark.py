# -*- coding:utf-8 -*-
"""
@author:Xu Jingyi
@file:Benchmark.py.py
@time:2018/12/1019:58
"""

import numpy as np
import random
import FactoryClass
import math

dataset = 'MNIST'
model_type = 'CNN'
seed = 10
# initialization = 'xavier'
model_architecture = [[32, 5, 5], [64, 5, 5], [1500]]
noise_level = 0
augmentation = False
dropout = 0.5
learning_rate = 0.001
batch_size = 128
epochs = 20
visualization_batch_num = 10


def run_benchmark():
    data_chooser = FactoryClass.ChooseDataset(dataset, seed, noise_level, augmentation)
    data_object = data_chooser.data_object
    x_train, y_train, x_test, y_test = data_object.x_train, data_object.y_train, data_object.x_test, data_object.y_test

    num_classes = data_object.num_classes
    input_size = data_object.input_size

    model_object = FactoryClass.ChooseNetworkCreator(model_type, model_architecture, input_size, learning_rate, dropout,
                                                     num_classes)
    classifier = model_object.choose_network_creator()

    record_file = 'benchmark_without_augmentation/' + dataset + '1.txt'
    record = open(record_file, 'a+')
    record.write('model architecture: ' + str(model_architecture) + '\n')
    record.write('noise level: ' + str(noise_level) + '\n')
    record.write('augmentation: ' + str(augmentation) + '\n')
    record.write('learning rate: ' + str(learning_rate) + '\n')
    record.write('batch size: ' + str(batch_size) + '\n')
    record.write('epoch: ' + str(epochs) + '\n')
    record.write('visualize after every ' + str(visualization_batch_num) + 'batch' + '\n')

    loss_train, accuracy_train = classifier.evaluate_model(x_train, y_train)
    loss_test, accuracy_test = classifier.evaluate_model(x_test, y_test)
    record.write('before training, train accuracy: ' + str(accuracy_train) +
                 ', test accuracy:' + str(accuracy_test) + '\n')
    record.flush()

    for epoch in range(epochs):
        num_samples = len(x_train)
        group_size = batch_size * visualization_batch_num
        num_group = math.ceil(num_samples / group_size)

        shuffle_index = np.arange(num_samples)
        random.shuffle(shuffle_index)
        x_train = x_train[shuffle_index]
        y_train = y_train[shuffle_index]
        for group in range(num_group):
            if group == (num_group - 1):
                index_subset = np.arange(group * group_size, num_samples)
            else:
                index_subset = np.arange(group * group_size, (group + 1) * group_size)
            classifier.train_model(x_train[index_subset], y_train[index_subset], batch_size, epochs=1)
            loss_train, accuracy_train = classifier.evaluate_model(x_train, y_train)
            loss_test, accuracy_test = classifier.evaluate_model(x_test, y_test)
            record.write(str(epoch) + '-th epoch, ' + str(group) + '-th group, loss: ' + str(
                loss_train) + ', train accuracy: ' + str(accuracy_train) + ', test accuracy:' + str(accuracy_test) + '\n')
            record.flush()
    record.write('*' * 30 + '\n')
    record.close()


run_benchmark()
