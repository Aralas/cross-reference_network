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
from keras import backend as K
import os
import gc
from keras.models import load_model
import copy
cfg = K.tf.ConfigProto()
cfg.gpu_options.allow_growth = True
K.set_session(K.tf.Session(config=cfg))

dataset = 'CIFAR10'
model_type = 'CNN'
seed = 10
# initialization = 'xavier'
model_architecture = [[30, 5, 5], [30, 5, 5], [30, 5, 5], [1000]]
noise_level = 0.8
augmentation = False
dropout = 0.5
learning_rate = 0.0002
batch_size = 200
section_num = 60
epochs = 1
data_size = 100
power_n = 4
lambda_weight = np.zeros(50)



# def randomly_sample_binary_data(x, y, data_size, label, index_clean):
#     indeces_positive = list(np.where(y[:, label] == 1)[0])
#     indeces_negative = set(range(len(y))) - set(indeces_positive)
#     index_train = [int(a) for a in index_clean[label]] + random.sample(indeces_negative, data_size)
#     print(len(index_train))
#     x_small = x[index_train]
#     y_small = np.array([1] * data_size + [0] * data_size).reshape(2 * data_size, 1)
#     shuffle_index = np.arange(len(x_small))
#     random.shuffle(shuffle_index)
#     x_small = x_small[shuffle_index]
#     y_small = y_small[shuffle_index]
#     return x_small, y_small


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


def generate_reference_output(x, binary_classifier_list, num_classes):
    num_sample = len(x)
    output = np.zeros((num_sample, num_classes))
    for i in range(num_classes):
        classifier = binary_classifier_list[i]
        prediction = classifier.prediction(x).reshape((num_sample,))
        output[:, i] = prediction
    return output


def run_cross_reference():
    data_chooser = FactoryClass.ChooseDataset(dataset, seed, noise_level, augmentation)
    data_object = data_chooser.data_object
    x_train, y_train, y_train_orig, x_test, y_test, clean_index = data_object.x_train, data_object.y_train, \
                                                                  data_object.y_train_orig, data_object.x_test, \
                                                                  data_object.y_test, data_object.clean_index
    num_classes = data_object.num_classes
    input_size = data_object.input_size

    binary_classifier_list = []
    model_object = FactoryClass.ChooseNetworkCreator(model_type, model_architecture, input_size, learning_rate, dropout,
                                                     2)

    dirs = 'test1/'
    if not os.path.exists(dirs):
        os.makedirs(dirs)

    model_dirs = dirs + 'model/' + dataset + '_RunTest5_1/'
    if not os.path.exists(model_dirs):
        os.makedirs(model_dirs)

    record_file = dirs + dataset + '_RunTest5_1.txt'
    record = open(record_file, 'a+')
    record.write('model architecture: ' + str(model_architecture) + '\n')
    record.write('noise level: ' + str(noise_level) + '\n')
    record.write('augmentation: ' + str(augmentation) + '\n')
    record.write('learning rate: ' + str(learning_rate) + '\n')
    record.write('batch size: ' + str(batch_size) + '\n')
    record.write('epoch: ' + str(epochs) + '\n')
    record.write('data size: ' + str(data_size) + '\n')
    record.write('lambda: ' + str(lambda_weight) + '\n')
    record.write('power n:' + str(power_n) + '\n')
    record.write('section: ' + str(section_num) + '\n')

    for label in range(num_classes):
        binary_classifier_list.append(model_object.choose_network_creator())

    # for top_n in range(1, 2):
    #     accuracy_multi = evaluate_target_model_top_n(x_test, y_test, binary_classifier_list, top_n)
    #     record.write('top ' + str(top_n) + ' test accuracy before training: ' + str(accuracy_multi) + '\n')
    #     record.flush()

    clean_x = x_train[clean_index]
    clean_y = y_train[clean_index]
    index_clean_matrix = np.zeros((num_classes, data_size))
    for label in range(num_classes):
        indeces_positive = list(np.where(clean_y[:, label] == 1)[0])
        index = random.sample(indeces_positive, data_size)
        index_clean_matrix[label] = index
    index_clean_matrix = index_clean_matrix.astype(int)

    for section in range(section_num):
        if section > 0 and section % 10 == 0:
            for label in range(num_classes):
                classifier = binary_classifier_list[label]
                classifier.model.save_weights(model_dirs + 'model' + str(label) + '.h5')
            del classifier
            del binary_classifier_list
            del reference_output
            gc.collect()

            binary_classifier_list = []
            for label in range(num_classes):
                classifier = model_object.choose_network_creator()
                classifier.model.load_weights(model_dirs + 'model' + str(label) + '.h5')
                print(classifier.prediction(x_train[0:1]))
                binary_classifier_list.append(classifier)

        for label in range(num_classes):
            print('section: ', section, 'label: ', label)
            classifier = binary_classifier_list[label]
            # x, y = randomly_sample_binary_data(x_train, y_train, data_size, label, index_clean_matrix)
            indeces_negative = list(np.where(y_train[:, label] != 1)[0])
            index_train = random.sample(indeces_negative, data_size)
            x = np.append(clean_x[index_clean_matrix[label]], x_train[index_train], axis=0)
            y = np.array([1] * data_size + [0] * data_size).reshape(2 * data_size, 1)
            shuffle_index = np.arange(len(x))
            random.shuffle(shuffle_index)
            x = x[shuffle_index]
            y = y[shuffle_index]

            classifier.power_n = power_n
            classifier.lamb_weight = lambda_weight[section]

            reference_output = generate_reference_output(x, binary_classifier_list, num_classes)

            np.delete(reference_output, label, axis=1)
            for batch in range(4):
                small_batch_size = len(x)//4
                small_data_index = range(small_batch_size * batch, small_batch_size * (batch + 1))
                # small_data_index = small_data_index.astype(int)
                classifier.reference_output = reference_output[small_data_index, :]

                classifier.train_model(x[small_data_index, :], y[small_data_index, :], small_batch_size, epochs)

        for top_n in range(1, 2):
            accuracy_multi = evaluate_target_model_top_n(x_test, y_test, binary_classifier_list, top_n)
            record.write(str(section) + '-th section, top ' + str(top_n) + ' test accuracy: ' + str(accuracy_multi) + '\n')
            record.flush()
    record.write('*' * 30 + '\n')
    record.close()


lambda_weight = [0] * section_num
for noise_level in [0.8]:
    run_cross_reference()

