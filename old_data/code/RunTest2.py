# -*- coding:utf-8 -*-
"""
@author:Xu Jingyi
@file:RunTest2.py.py
@time:2018/12/2804:06
"""

import tensorflow as tf
import numpy as np
import random
import FactoryClass


dataset = 'MNIST'
model_type = 'CNN'
seed = 10
# initialization = 'xavier'
model_architecture = [[32, 5, 5], [64, 5, 5], [1500]]
noise_level = 1
augmentation = True
dropout = 0.5
learning_rate = 0.001
batch_size = 128
section_num = 20
epochs_for_binary = 5
epochs_for_multiple = 1
data_size = 1000


# first_merged_section = 5
# first_update_section = 50
# first_value_to_label_section = 10
# update_threshold = [0.8, 0.6]


def randomly_sample_binary_data(x, y, data_size, label):
    indeces_positive = list(np.where(y[:, label] == 1)[0])
    indeces_negative = set(range(len(y))) - set(indeces_positive)
    index_train = random.sample(indeces_positive, data_size) + random.sample(indeces_negative, data_size)
    x_small = x[index_train]
    y_small = np.array([1] * data_size + [0] * data_size).reshape(2 * data_size, 1)
    shuffle_index = np.arange(len(x_small))
    random.shuffle(shuffle_index)
    x_small = x_small[shuffle_index]
    y_small = y_small[shuffle_index]
    return x_small, y_small


def multi_label_to_binary_label(y, label):
    y_hat = np.zeros((len(y), 1))
    indeces_positive = list(np.where(y[:, label] == 1)[0])
    y_hat[indeces_positive] = 1
    return y_hat


def generate_probability_matrix(x, binary_classifier_list):
    num_classes = len(binary_classifier_list)
    num_sample = len(x)
    result = np.zeros((num_sample, num_classes))
    for label in range(num_classes):
        classifier = binary_classifier_list[label]
        prediction = classifier.prediction(x).reshape((num_sample,))
        result[:, label] = prediction
    with tf.Session() as sess:
        probability_matrix = sess.run(tf.nn.softmax(result))
    return probability_matrix


def run_cross_reference():
    data_chooser = FactoryClass.ChooseDataset(dataset, seed, noise_level, augmentation)
    data_object = data_chooser.data_object
    x_train, y_train, y_train_orig, x_test, y_test = data_object.x_train, data_object.y_train, \
                                                     data_object.y_train_orig, data_object.x_test, data_object.y_test

    num_classes = data_object.num_classes
    input_size = data_object.input_size

    binary_classifier_list = []
    binary_model_object = FactoryClass.ChooseNetworkCreator(model_type, model_architecture, input_size, learning_rate,
                                                            dropout, 2)
    multi_model_object = FactoryClass.ChooseNetworkCreator(model_type, model_architecture, input_size, learning_rate,
                                                           dropout, num_classes)
    record_file = 'test2_1/' + dataset + '.txt'
    record = open(record_file, 'a+')
    record.write('model architecture: ' + str(model_architecture) + '\n')
    record.write('noise level: ' + str(noise_level) + '\n')
    record.write('augmentation: ' + str(augmentation) + '\n')
    record.write('learning rate: ' + str(learning_rate) + '\n')
    record.write('batch size: ' + str(batch_size) + '\n')
    record.write('epoch for binary classifiers: ' + str(epochs_for_binary) + ', multi-classifier: ' + str(
        epochs_for_multiple) + '\n')
    record.write('data size: ' + str(data_size) + '\n')
    record.write('section: ' + str(section_num) + '\n')

    for label in range(num_classes):
        binary_classifier_list.append(binary_model_object.choose_network_creator())
    multi_classifier = multi_model_object.choose_network_creator()
    for section in range(section_num):

        for label in range(num_classes):
            classifier = binary_classifier_list[label]
            x, y = randomly_sample_binary_data(x_train, y_train, data_size, label)
            classifier.train_model(x, y, batch_size, epochs_for_binary)
            loss_train, accuracy_train = classifier.evaluate_model(x, y)
            record.write(str(section) + '-th section, ' + str(label) + '-th classifier, loss: ' + str(loss_train)
                         + ', train accuracy: ' + str(accuracy_train) + '\n')
            record.flush()
        for epoch in range(epochs_for_multiple):
            new_y = generate_probability_matrix(x_train, binary_classifier_list)
            multi_classifier.train_model(x_train, new_y, batch_size, epochs=1)
            loss_test, accuracy_test = multi_classifier.evaluate_model(x_test, y_test)
            record.write(str(section) + '-th section, ' + str(epoch) + '-th epoch, test accuracy:' + str(
                accuracy_test) + '\n')
            record.flush()
    record.write('*' * 30 + '\n')
    record.close()


for noise_level in [0.5]:
    run_cross_reference()
