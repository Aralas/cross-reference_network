# -*- coding:utf-8 -*-
"""
@author:Xu Jingyi
@file:Readtxt.txt.py
@time:2019/1/201:28
"""


import sys
import matplotlib.pyplot as plt
import numpy as np


def read_file(filename):
    file = open(filename)
    lines = file.readlines()
    train_acc = []
    test_acc = []

    for line in lines:
        line = line.strip().split(',')
        if len(line) == 3:
            train_acc.append(float(line[1].split(':')[1]))
            test_acc.append(float(line[2].split(':')[1]))
        elif len(line) == 5:
            train_acc.append(float(line[3].split(':')[1]))
            test_acc.append(float(line[4].split(':')[1]))

    return train_acc, test_acc



fig, axes = plt.subplots(nrows=3, ncols=3)
x_position = [0]
x_value = np.arange(21)
for epoch in range(20):
    x_position.append((epoch + 1) * 94 + 1)
plt.setp(axes, xticks=x_position, xticklabels=x_value)
# plt.suptitle('Benchmark of MNIST with small model without augmentation, learning rate=0.001, {[5, 5*5], [5, 5*5], [200]}')
# plt.suptitle('Benchmark of MNIST with small model without augmentation, learning rate=0.001, {[5, 5*5], [5, 5*5], [200]}')
plt.suptitle('Benchmark of MNIST, learning rate=0.001, {[32, 5*5], [64, 5*5], [1500]}')
# plt.suptitle('Benchmark of MNIST without augmentation, learning rate=0.001, {[32, 5*5], [64, 5*5], [1500]}')



for i in range(1, 10):
    if i != 8:
        filename = 'benchmark1_without_shuffle/MNIST' + str(i-1) + '.txt'
        # filename = 'benchmark_without_augmentation_without_shuffle/MNIST' + str(i) + '.txt'
        # filename = 'benchmark_small_architecture_without_augmentation_without_shuffle/MNIST' + str(i) + '.txt'
        # filename = 'benchmark_small_architecture_without_augmentation_without_shuffle/MNIST' + str(i) + '.txt'
        train_acc, test_acc = read_file(filename)
        x_axis = np.arange(len(train_acc))
        plt.subplot(3, 3, i)

        plt.plot(x_axis, train_acc, color='blue', linewidth=2.0, label='train')
        plt.plot(x_axis, test_acc, color='red', linewidth=2.0, label='test')
        plt.legend(loc='best')
        plt.xticks(x_position, x_value)
        plt.ylim((0, 1))
        # plt.xlim((0, 20 * 94 + 1))
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.title('noise level:'+str((i-1)/10))

# plt.tight_layout()
plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.show()