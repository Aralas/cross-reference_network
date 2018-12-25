# -*- coding:utf-8 -*-
"""
@author:Xu Jingyi
@file:RunTest.py
@time:2018/11/2722:28
"""

import tensorflow as tf
import numpy as np
import random
import FactoryClass
from copy import deepcopy

dataset = 'MNIST'
model_type = 'CNN'
seed = 10
# initialization = 'xavier'
model_architecture = [[32, 5, 5], [32, 5, 5], [1000]]
noise_level = 1
augmentation = True
dropout = 0.5
learning_rate = 0.001
batch_size = 128
section_num = 50
epochs = 2
data_size = 1000
first_merged_section = 5
update_threshold = [0.8, 0.6]
first_update_section = 10


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


def generate_combined_label(x, y, label, num_classes, binary_classifier_list):
    n = len(x)
    result = np.zeros((num_classes, n))
    for i in range(num_classes):
        classifier = binary_classifier_list[i]
        prediction = 1 - classifier.prediction(x).reshape((n,))
        result[i, :] = prediction
    result[label, :] = y.reshape(n, )
    factor = np.ones((num_classes, 1)) / (2 * (num_classes - 1))
    factor[label] = 0.5
    y_hat = result * factor
    y_hat = np.sum(y_hat, axis=0).reshape((n, 1))
    return y_hat


def multi_label_to_binary_label(y, label):
    y_hat = np.zeros((len(y), 1))
    indeces_positive = list(np.where(y[:, label] == 1)[0])
    y_hat[indeces_positive] = 1
    return y_hat


def evaluate_target_model_top_n(x, y, binary_classifier_list, top_n):
    num_classes = len(binary_classifier_list)
    num_sample = len(x)
    result = np.zeros((num_sample, num_classes))
    for label in range(num_classes):
        classifier = binary_classifier_list[label]
        prediction = classifier.prediction(x).reshape((num_sample,))
        result[:, label] = prediction
    # result = np.argmax(result, axis=1)
    # accuracy = np.mean(np.argmax(y, axis=1) == np.argmax(result, axis=1))
    with tf.Session() as sess:
        accuracy = sess.run(tf.reduce_mean(
            tf.cast(tf.nn.in_top_k(predictions=result, targets=np.argmax(y, axis=1), k=top_n), tf.float32)))
    return accuracy


def update_label(x, y, y_orig, update_threshold, binary_classifier_list):
    beta1, beta2 = update_threshold
    num_classes = len(binary_classifier_list)
    num_sample = len(x)
    result = np.zeros((num_sample, num_classes))
    y_new = deepcopy(y)

    for label in range(num_classes):
        classifier = binary_classifier_list[label]
        prediction = classifier.prediction(x).reshape((num_sample,))
        result[:, label] = prediction

    false_label_index = np.where(y != y_orig)[0]
    predict_label_thres1 = 1 * (result > beta1)
    predict_label_thres2 = 1 * (result > beta2)
    count_prediction1 = np.sum(predict_label_thres1, axis=1)
    count_prediction2 = np.sum(predict_label_thres2, axis=1)
    confident_prediction_index = np.where((count_prediction1 == 1) & (count_prediction2 == 1))[0]
    y_new[confident_prediction_index] = np.argmax(result[confident_prediction_index], axis=1).reshape(-1, 1)
    false_predict_index = np.where(y_new != y_orig)[0]
    n1 = len(set(false_label_index) - set(false_predict_index))
    n2 = len(set(false_predict_index) - set(false_label_index))

    return y_new, n1, n2


def run_cross_reference():
    data_chooser = FactoryClass.ChooseDataset(dataset, seed, noise_level, augmentation)
    data_object = data_chooser.data_object
    x_train, y_train, y_train_orig, x_test, y_test = data_object.x_train, data_object.y_train, \
                                                     data_object.y_train_orig, data_object.x_test, data_object.y_test
    y_train_noise = deepcopy(y_train)
    false_label_num = np.sum(y_train != y_train_orig)
    num_classes = data_object.num_classes
    input_size = data_object.input_size

    binary_classifier_list = []
    model_object = FactoryClass.ChooseNetworkCreator(model_type, model_architecture, input_size, learning_rate, dropout,
                                                     2)
    record_file = 'test2/' + dataset + '.txt'
    record = open(record_file, 'a+')
    record.write('model architecture: ' + str(model_architecture) + '\n')
    record.write('noise level: ' + str(noise_level) + '\n')
    record.write('augmentation: ' + str(augmentation) + '\n')
    record.write('learning rate: ' + str(learning_rate) + '\n')
    record.write('batch size: ' + str(batch_size) + '\n')
    record.write('epoch: ' + str(epochs) + '\n')
    record.write('data size: ' + str(data_size) + '\n')
    record.write('first merged section: ' + str(first_merged_section) + '\n')
    record.write('section: ' + str(section_num) + '\n')
    record.write('update threshold: ' + str(update_threshold) + '\n')
    record.write('first update section: ' + str(first_update_section) + '\n')

    for label in range(num_classes):
        binary_classifier_list.append(model_object.choose_network_creator())

    for top_n in range(1, 4):
        accuracy_multi = evaluate_target_model_top_n(x_test, y_test, binary_classifier_list, top_n)
        record.write('top ' + str(top_n) + ' test accuracy before training: ' + str(accuracy_multi) + '\n')
        record.flush()

    for section in range(section_num):
        if section >= first_update_section:
            y_train, n1, n2 = update_label(x_train, y_train_noise, y_train_orig, update_threshold,
                                                         binary_classifier_list)
        record.write('successfully update noise label: ' + str(n1) + ' false update: ' + str(n2) + '\n')
        record.flush()
        for label in range(num_classes):
            classifier = binary_classifier_list[label]
            x, y = randomly_sample_binary_data(x_train, y_train, data_size, label)
            if section >= first_merged_section:
                y = generate_combined_label(x, y, label, num_classes, binary_classifier_list)
            classifier.train_model(x, y, batch_size, epochs)
            loss_train, accuracy_train = classifier.evaluate_model(x, y)
            loss_test, accuracy_test = classifier.evaluate_model(x_test, multi_label_to_binary_label(y_test, label))
            record.write(str(section) + '-th section, ' + str(label) + '-th classifier, loss: ' + str(loss_train)
                         + ', train accuracy: ' + str(accuracy_train) + ', test accuracy:' + str(accuracy_test) + '\n')
            record.flush()
        for top_n in range(1, 4):
            accuracy_multi = evaluate_target_model_top_n(x_test, y_test, binary_classifier_list, top_n)
            record.write('top ' + str(top_n) + ' test accuracy: ' + str(accuracy_multi) + '\n')
            record.flush()
    record.write('*' * 30 + '\n')
    record.close()


for noise_level in [0.5]:
    run_cross_reference()
