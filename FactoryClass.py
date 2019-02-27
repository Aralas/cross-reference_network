# -*- coding:utf-8 -*-
"""
@author:Xu Jingyi
@file:FactoryClass.py
@time:2018/11/2817:14
"""

import CreateNetwork
import DataPreProcessing


class ChooseNetworkCreator(object):

    def __init__(self, model_type, architecture, input_shape, learning_rate, dropout, num_classes):
        self.model_type = model_type
        self.architecture = architecture
        self.input_shape = input_shape
        self.dropout = dropout
        self.num_classes = num_classes
        self.learning_rate = learning_rate

    def choose_network_creator(self):
        if self.model_type == 'CNN':
            return CreateNetwork.CreateCNN(self.architecture, self.input_shape, self.learning_rate, self.dropout,
                                           self.num_classes)
        elif self.model_type == 'fully_connected':
            return CreateNetwork.CreateFullyConnected(self.architecture, self.input_shape, self.learning_rate,
                                                      self.dropout, self.num_classes)
        else:
            print('There is no such network /(ㄒoㄒ)/~~')


class ChooseDataset(object):
    def __init__(self, dataset, seed, noise_level, augmentation):
        self.dataset = dataset
        self.seed = seed
        self.noise_level = noise_level
        self.augmentation = augmentation
        self.data_object = self.choose_dataset()

    def choose_dataset(self):
        if self.dataset == 'MNIST':
            return DataPreProcessing.MNIST(self.seed, self.noise_level, self.augmentation)
        elif self.dataset == 'CIFAR10':
            return DataPreProcessing.CIFAR10(self.seed, self.noise_level, self.augmentation)
        elif self.dataset == 'Fruit360':
            return DataPreProcessing.Fruit360(self.seed, self.noise_level, self.augmentation)
        elif self.dataset == 'NORB':
            return DataPreProcessing.NORB(self.seed, self.noise_level, self.augmentation)
        elif self.dataset == 'NewsGroup':
            return DataPreProcessing.NewsGroup(self.seed, self.noise_level, self.augmentation)
        else:
            print('There is no such dataset /(ㄒoㄒ)/~~')
