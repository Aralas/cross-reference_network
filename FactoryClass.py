# -*- coding:utf-8 -*-
"""
@author:Xu Jingyi
@file:FactoryClass.py
@time:2018/11/2817:14
"""

import CreateNetwork
import DataPreProcessing

class ChooseNetworkCreator(object):

    def __init__(self, model_type, architecture, input_shape, dropout, num_classes):
        self.model_type = model_type
        self.architecture = architecture
        self.input_shape = input_shape
        self.dropout = dropout
        self.num_classes = num_classes

    def choose_network_creator(self):
        if self.model_type == 'CNN':
            return CreateNetwork.CreateCNN(architecture, self.input_shape, self.dropout, self.num_classes)
        elif self.model_type == 'fully_connected':
            return CreateNetwork.CreateFullyConnected(architecture, self.input_shape, self.dropout, self.num_classes)
        else:
            print('There is no such network /(ㄒoㄒ)/~~')


class ChooseDataset(object):
    def __init__(self, dataset, seed, noise_level, augmentation):
        self.dataset = dataset
        self.parameters = seed, noise_level, augmentation

    def choose_dataset(self):
        if self.dataset == 'MNIST':
            return DataPreProcessing.MNIST(self.parameters)
        elif self.dataset == 'CIFAR10':
            return DataPreProcessing.CIFAR10(self.parameters)
        elif self.dataset == 'NORB':
            return DataPreProcessing.NORB(self.parameters)
        elif self.dataset == 'NewsGroup':
            return DataPreProcessing.NewsGroup(self.parameters)
        else:
            print('There is no such dataset /(ㄒoㄒ)/~~')


