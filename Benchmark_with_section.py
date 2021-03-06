# -*- coding:utf-8 -*-
"""
@author:Xu Jingyi
@file:Benchmark.py.py
@time:2018/12/1019:58
"""

import numpy as np
import random
import FactoryClass

dataset = 'MNIST'
model_type = 'CNN'
seed = 10
# initialization = 'xavier'
model_architecture = [[32, 5, 5], [64, 5, 5], [1500]]
noise_level = 0.5
augmentation = True
dropout = 0.5
learning_rate = 0.001
batch_size = 128
epochs_for_each_section = 20
data_size = 20000
section_num = 20


def run_benchmark():
    data_chooser = FactoryClass.ChooseDataset(dataset, seed, noise_level, augmentation)
    data_object = data_chooser.data_object
    x_train, y_train, x_test, y_test = data_object.x_train, data_object.y_train, data_object.x_test, data_object.y_test

    num_classes = data_object.num_classes
    input_size = data_object.input_size

    model_object = FactoryClass.ChooseNetworkCreator(model_type, model_architecture, input_size, learning_rate, dropout,
                                                     num_classes)
    classifier = model_object.choose_network_creator()

    record_file = 'benchmark_with_section/' + dataset + '.txt'
    record = open(record_file, 'a+')
    record.write('model architecture: ' + str(model_architecture) + '\n')
    record.write('noise level: ' + str(noise_level) + '\n')
    record.write('augmentation: ' + str(augmentation) + '\n')
    record.write('learning rate: ' + str(learning_rate) + '\n')
    record.write('batch size: ' + str(batch_size) + '\n')
    record.write('data size: ' + str(data_size) + '\n')
    record.write('epoch for each section: ' + str(epochs_for_each_section) + '\n')

    for section in range(section_num):

        index_train = random.sample(range(len(y_train)), data_size)
        x_small = x_train[index_train]
        y_small = y_train[index_train]
        for epoch in range(epochs_for_each_section):
            classifier.train_model(x_small, y_small, batch_size, epochs=1)
            loss_train, accuracy_train = classifier.evaluate_model(x_train, y_train)
            loss_test, accuracy_test = classifier.evaluate_model(x_test, y_test)
            record.write(str(section) + '-th section, ' + str(epoch) + '-th epoch, loss: ' + str(
                loss_train) + ', train accuracy: ' + str(accuracy_train) + ', test accuracy:' + str(
                accuracy_test) + '\n')
            record.flush()
    record.write('*' * 30 + '\n')
    record.close()


run_benchmark()
