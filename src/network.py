'''
Created on May 17, 2019

@author: Fernando Moya Rueda, Erik Altermann, Arthur Matei
@email: erik.altermann@tu-dortmund.de, 	fernando.moya@tu-dortmund.de, arthur.matei@tu-dortmund.de
'''

from __future__ import print_function
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import matplotlib.pyplot as plt
import numpy as np
from tpp import TPP
import math
import pickle
from settings import window_size

config = {'dataset': 'mbientlab100', 'network': 'cnn_imu_tpp',
          'output': 'attribute', 'num_filters': 64, 'filter_size': 5,
          'folder_exp': '',
          'NB_sensor_channels': 30,
          'sliding_window_length': 200, 'num_attributes': 19, 'num_classes': 7,
          'reshape_input': False,
          'aggregate': 'FC', 'pooling': 0,
          'storing_acts': False}


class Network(nn.Module):
    '''
    classdocs
    '''

    def __init__(self, config):
        '''
        Constructor
        '''

        super(Network, self).__init__()

        logging.info('            Network: Constructor')

        self.config = config
        self.tpp = TPP(config)
        self.BN_bool = False
        self.limbs3 = False

        if self.config["reshape_input"]:
            in_channels = 3
            Hx = int(self.config['NB_sensor_channels'] / 3)
        else:
            in_channels = 1
            Hx = self.config['NB_sensor_channels']
        Wx = self.config['sliding_window_length']

        if self.config["dataset"] in ['mbientlab_quarter', 'gesture', 'mocap_quarter', 'virtual_quarter'] \
                and self.config["pooling"] in [1, 2, 3, 4]:
            padding = (2, 0)
        else:
            if self.config["aggregate"] in ["FCN", "LSTM"]:
                padding = (2, 0)
            elif self.config["aggregate"] == "FC":
                padding = 0

        # Computing the size of the feature maps
        Wx, Hx = self.size_feature_map(Wx=Wx,
                                       Hx=Hx,
                                       F=(self.config['filter_size'], 1),
                                       P=padding, S=(1, 1), type_layer='conv')
        logging.info('            Network: Wx {} and Hx {}'.format(Wx, Hx))
        Wx, Hx = self.size_feature_map(Wx=Wx,
                                       Hx=Hx,
                                       F=(self.config['filter_size'], 1),
                                       P=padding, S=(1, 1), type_layer='conv')
        logging.info('            Network: Wx {} and Hx {}'.format(Wx, Hx))
        if self.config["pooling"] in [1, 2, 3, 4]:
            if self.config["pooling"] in [1, 2]:
                Wx = int(Wx / 2) - 1
            else:
                if self.config["aggregate"] in ["FCN", "LSTM"] or self.config["dataset"] in ['mbientlab_quarter',
                                                                                             'gesture', 'mocap_quarter',
                                                                                             'virtual_quarter']:
                    Wx, Hx = self.size_feature_map(Wx=Wx,
                                                   Hx=Hx,
                                                   F=(3, 1),
                                                   P=(1, 0), S=(1, 1), type_layer='pool')
                else:
                    Wx, Hx = self.size_feature_map(Wx=Wx,
                                                   Hx=Hx,
                                                   F=(2, 1),
                                                   P=padding, S=(2, 1), type_layer='pool')
            Wxp1 = Wx
            self.pooling_Wx = [Wxp1]
            logging.info('            Network: Wx {} and Hx {}'.format(Wx, Hx))
        Wx, Hx = self.size_feature_map(Wx=Wx,
                                       Hx=Hx,
                                       F=(self.config['filter_size'], 1),
                                       P=padding, S=(1, 1), type_layer='conv')
        logging.info('            Network: Wx {} and Hx {}'.format(Wx, Hx))
        Wx, Hx = self.size_feature_map(Wx=Wx,
                                       Hx=Hx,
                                       F=(self.config['filter_size'], 1),
                                       P=padding, S=(1, 1), type_layer='conv')
        logging.info('            Network: Wx {} and Hx {}'.format(Wx, Hx))
        if self.config["pooling"] in [2, 4]:
            if self.config["pooling"] in [2]:
                Wx = int(Wx / 2) - 1
            else:
                if self.config["aggregate"] in ["FCN", "LSTM"] or self.config["dataset"] in ['mbientlab_quarter',
                                                                                             'gesture', 'mocap_quarter',
                                                                                             'virtual_quarter']:
                    Wx, Hx = self.size_feature_map(Wx=Wx,
                                                   Hx=Hx,
                                                   F=(3, 1),
                                                   P=(1, 0), S=(1, 1), type_layer='pool')
                else:
                    Wx, Hx = self.size_feature_map(Wx=Wx,
                                                   Hx=Hx,
                                                   F=(2, 1),
                                                   P=padding, S=(2, 1), type_layer='pool')
            Wxp2 = Wx
            self.pooling_Wx = [Wxp1, Wxp2]
            logging.info('            Network: Wx {} and Hx {}'.format(Wx, Hx))

        # set the Conv layers
        if self.config["network"] in ["cnn", "cnn_tpp"]:
            self.conv1_1 = nn.Conv2d(in_channels=in_channels,
                                     out_channels=self.config['num_filters'],
                                     kernel_size=(self.config['filter_size'], 1),
                                     stride=1, padding=padding)
            self.conv1_2 = nn.Conv2d(in_channels=self.config['num_filters'],
                                     out_channels=self.config['num_filters'],
                                     kernel_size=(self.config['filter_size'], 1),
                                     stride=1, padding=padding)
            self.conv2_1 = nn.Conv2d(in_channels=self.config['num_filters'],
                                     out_channels=self.config['num_filters'],
                                     kernel_size=(self.config['filter_size'], 1),
                                     stride=1, padding=padding)
            self.conv2_2 = nn.Conv2d(in_channels=self.config['num_filters'],
                                     out_channels=self.config['num_filters'],
                                     kernel_size=(self.config['filter_size'], 1),
                                     stride=1, padding=padding)

            if self.config["network"] in ["cnn_tpp"] and self.BN_bool:
                self.BN = nn.BatchNorm2d(self.config['num_filters'])

            if self.config["aggregate"] == "FCN":
                if self.config["reshape_input"]:
                    self.fc3 = nn.Conv2d(in_channels=self.config['num_filters'],
                                         out_channels=256, kernel_size=(1, 1), stride=1, padding=0)
                else:
                    self.fc3 = nn.Conv2d(in_channels=self.config['num_filters'],
                                         out_channels=256, kernel_size=(1, 1), stride=1, padding=0)
            elif self.config["aggregate"] == "FC":
                if self.config["network"] == "cnn_tpp":
                    self.fc3 = nn.Linear(self.tpp.get_output(), 256)
                else:
                    if self.config["reshape_input"]:
                        self.fc3 = nn.Linear(self.config['num_filters'] *
                                             int(Wx) * int(self.config['NB_sensor_channels'] / 3), 256)
                    else:
                        self.fc3 = nn.Linear(self.config['num_filters'] * int(Wx) * self.config['NB_sensor_channels'],
                                             256)
            elif self.config["aggregate"] == "LSTM":
                if self.config["reshape_input"]:
                    self.fc3 = nn.LSTM(input_size=self.config['num_filters'] *
                                                  int(self.config['NB_sensor_channels'] / 3), hidden_size=256,
                                       batch_first=True, bidirectional=True)
                else:
                    self.fc3 = nn.LSTM(input_size=self.config['num_filters'] * self.config['NB_sensor_channels'],
                                       hidden_size=256, batch_first=True, bidirectional=True)

        # set the Conv layers
        if self.config["network"] in ["cnn_imu", "cnn_imu_tpp"]:
            # LA
            self.conv_LA_1_1 = nn.Conv2d(in_channels=in_channels,
                                         out_channels=self.config['num_filters'],
                                         kernel_size=(self.config['filter_size'], 1),
                                         stride=1, padding=padding)

            self.conv_LA_1_2 = nn.Conv2d(in_channels=self.config['num_filters'],
                                         out_channels=self.config['num_filters'],
                                         kernel_size=(self.config['filter_size'], 1),
                                         stride=1, padding=padding)

            self.conv_LA_2_1 = nn.Conv2d(in_channels=self.config['num_filters'],
                                         out_channels=self.config['num_filters'],
                                         kernel_size=(self.config['filter_size'], 1),
                                         stride=1, padding=padding)

            self.conv_LA_2_2 = nn.Conv2d(in_channels=self.config['num_filters'],
                                         out_channels=self.config['num_filters'],
                                         kernel_size=(self.config['filter_size'], 1),
                                         stride=1, padding=padding)

            if self.BN_bool:
                self.BN_LA = nn.BatchNorm2d(self.config['num_filters'])

            if self.config["aggregate"] == "FCN":
                self.fc3_LA = nn.Conv2d(in_channels=self.config['num_filters'],
                                        out_channels=256, kernel_size=(1, 1), stride=1, padding=0)
            elif self.config["aggregate"] == "LSTM":
                if self.config["reshape_input"]:
                    if self.config["NB_sensor_channels"] == 27:
                        self.fc3_LA = nn.LSTM(input_size=self.config['num_filters'] *
                                                         int(self.config['NB_sensor_channels'] / 9),
                                              hidden_size=256, batch_first=True, bidirectional=True)
                    elif self.config["NB_sensor_channels"] == 30:
                        self.fc3_LA = nn.LSTM(input_size=self.config['num_filters'] *
                                                         int(self.config['NB_sensor_channels'] / 15),
                                              hidden_size=256, batch_first=True, bidirectional=True)
                    elif self.config["NB_sensor_channels"] == 126:
                        self.fc3_LA = nn.LSTM(input_size=self.config['num_filters'] * 10,
                                              hidden_size=256, batch_first=True, bidirectional=True)
                else:
                    if self.config["NB_sensor_channels"] == 27:
                        self.fc3_LA = nn.LSTM(input_size=self.config['num_filters'] *
                                                         int(self.config['NB_sensor_channels'] / 3),
                                              hidden_size=256, batch_first=True, bidirectional=True)
                    elif self.config["NB_sensor_channels"] == 30:
                        self.fc3_LA = nn.LSTM(input_size=self.config['num_filters'] *
                                                         int(self.config['NB_sensor_channels'] / 5),
                                              hidden_size=256, batch_first=True, bidirectional=True)
                    elif self.config["NB_sensor_channels"] == 126:
                        self.fc3_LA = nn.LSTM(input_size=self.config['num_filters'] * 30,
                                              hidden_size=256, batch_first=True, bidirectional=True)
            elif self.config["aggregate"] == "FC":
                if self.config["network"] == "cnn_imu_tpp":
                    self.fc3_LA = nn.Linear(self.tpp.get_output(), 256)
                else:
                    if self.config["reshape_input"]:
                        if self.config["NB_sensor_channels"] == 27:
                            self.fc3_LA = nn.Linear(self.config['num_filters'] * int(Wx) *
                                                    int(self.config['NB_sensor_channels'] / 9), 256)
                        elif self.config["NB_sensor_channels"] == 30:
                            self.fc3_LA = nn.Linear(self.config['num_filters'] * int(Wx) *
                                                    int(self.config['NB_sensor_channels'] / 15), 256)
                        elif self.config["NB_sensor_channels"] == 126:
                            self.fc3_LA = nn.Linear(self.config['num_filters'] * int(Wx) * 10, 256)
                    else:
                        if self.config["NB_sensor_channels"] == 27:
                            self.fc3_LA = nn.Linear(self.config['num_filters'] * int(Wx) *
                                                    int(self.config['NB_sensor_channels'] / 3), 256)
                        elif self.config["NB_sensor_channels"] == 30:
                            self.fc3_LA = nn.Linear(self.config['num_filters'] * int(Wx) *
                                                    int(self.config['NB_sensor_channels'] / 5), 256)
                        elif self.config["NB_sensor_channels"] == 126:
                            self.fc3_LA = nn.Linear(self.config['num_filters'] * int(Wx) * 30, 256)

            # LL
            self.conv_LL_1_1 = nn.Conv2d(in_channels=in_channels,
                                         out_channels=self.config['num_filters'],
                                         kernel_size=(self.config['filter_size'], 1),
                                         stride=1, padding=padding)

            self.conv_LL_1_2 = nn.Conv2d(in_channels=self.config['num_filters'],
                                         out_channels=self.config['num_filters'],
                                         kernel_size=(self.config['filter_size'], 1),
                                         stride=1, padding=padding)

            self.conv_LL_2_1 = nn.Conv2d(in_channels=self.config['num_filters'],
                                         out_channels=self.config['num_filters'],
                                         kernel_size=(self.config['filter_size'], 1),
                                         stride=1, padding=padding)

            self.conv_LL_2_2 = nn.Conv2d(in_channels=self.config['num_filters'],
                                         out_channels=self.config['num_filters'],
                                         kernel_size=(self.config['filter_size'], 1),
                                         stride=1, padding=padding)

            if self.BN_bool:
                self.BN_LL = nn.BatchNorm2d(self.config['num_filters'])

            if self.config["aggregate"] == "FCN":
                self.fc3_LL = nn.Conv2d(in_channels=self.config['num_filters'],
                                        out_channels=256, kernel_size=(1, 1), stride=1, padding=0)
            elif self.config["aggregate"] == "LSTM":
                if self.config["reshape_input"]:
                    if self.config["NB_sensor_channels"] == 30:
                        self.fc3_LL = nn.LSTM(input_size=self.config['num_filters'] *
                                                         int(self.config['NB_sensor_channels'] / 15),
                                              hidden_size=256, batch_first=True, bidirectional=True)
                    elif self.config["NB_sensor_channels"] == 126:
                        self.fc3_LL = nn.LSTM(input_size=self.config['num_filters'] * 8,
                                              hidden_size=256, batch_first=True, bidirectional=True)
                else:
                    if self.config["NB_sensor_channels"] == 30:
                        self.fc3_LL = nn.LSTM(input_size=self.config['num_filters'] *
                                                         int(self.config['NB_sensor_channels'] / 5),
                                              hidden_size=256, batch_first=True, bidirectional=True)
                    elif self.config["NB_sensor_channels"] == 126:
                        self.fc3_LL = nn.LSTM(input_size=self.config['num_filters'] * 24,
                                              hidden_size=256, batch_first=True, bidirectional=True)
            elif self.config["aggregate"] == "FC":
                if self.config["network"] == "cnn_imu_tpp":
                    self.fc3_LL = nn.Linear(self.tpp.get_output(), 256)
                else:
                    if self.config["reshape_input"]:
                        if self.config["NB_sensor_channels"] == 30:
                            self.fc3_LL = nn.Linear(self.config['num_filters'] * int(Wx) *
                                                    int(self.config['NB_sensor_channels'] / 15), 256)
                        elif self.config["NB_sensor_channels"] == 126:
                            self.fc3_LL = nn.Linear(self.config['num_filters'] * int(Wx) * 8, 256)
                    else:
                        if self.config["NB_sensor_channels"] == 30:
                            self.fc3_LL = nn.Linear(self.config['num_filters'] * int(Wx) *
                                                    int(self.config['NB_sensor_channels'] / 5), 256)
                        elif self.config["NB_sensor_channels"] == 126:
                            self.fc3_LL = nn.Linear(self.config['num_filters'] * int(Wx) * 24, 256)

            # N
            self.conv_N_1_1 = nn.Conv2d(in_channels=in_channels,
                                        out_channels=self.config['num_filters'],
                                        kernel_size=(self.config['filter_size'], 1),
                                        stride=1, padding=padding)

            self.conv_N_1_2 = nn.Conv2d(in_channels=self.config['num_filters'],
                                        out_channels=self.config['num_filters'],
                                        kernel_size=(self.config['filter_size'], 1),
                                        stride=1, padding=padding)

            self.conv_N_2_1 = nn.Conv2d(in_channels=self.config['num_filters'],
                                        out_channels=self.config['num_filters'],
                                        kernel_size=(self.config['filter_size'], 1),
                                        stride=1, padding=padding)

            self.conv_N_2_2 = nn.Conv2d(in_channels=self.config['num_filters'],
                                        out_channels=self.config['num_filters'],
                                        kernel_size=(self.config['filter_size'], 1),
                                        stride=1, padding=padding)

            if self.BN_bool:
                self.BN_N = nn.BatchNorm2d(self.config['num_filters'])

            if self.config["aggregate"] == "FCN":
                self.fc3_N = nn.Conv2d(in_channels=self.config['num_filters'],
                                       out_channels=256, kernel_size=(1, 1), stride=1, padding=0)
            elif self.config["aggregate"] == "LSTM":
                if self.config["reshape_input"]:
                    if self.config["NB_sensor_channels"] == 27:
                        self.fc3_N = nn.LSTM(input_size=self.config['num_filters'] *
                                                        int(self.config['NB_sensor_channels'] / 9),
                                             hidden_size=256, batch_first=True, bidirectional=True)
                    elif self.config["NB_sensor_channels"] == 30:
                        self.fc3_N = nn.LSTM(input_size=self.config['num_filters'] *
                                                        int(self.config['NB_sensor_channels'] / 15),
                                             hidden_size=256, batch_first=True, bidirectional=True)
                    elif self.config["NB_sensor_channels"] == 126:
                        self.fc3_N = nn.LSTM(input_size=self.config['num_filters'] * 6,
                                             hidden_size=256, batch_first=True, bidirectional=True)
                else:
                    if self.config["NB_sensor_channels"] == 27:
                        self.fc3_N = nn.LSTM(input_size=self.config['num_filters'] *
                                                        int(self.config['NB_sensor_channels'] / 3),
                                             hidden_size=256, batch_first=True, bidirectional=True)
                    elif self.config["NB_sensor_channels"] == 30:
                        self.fc3_N = nn.LSTM(input_size=self.config['num_filters'] *
                                                        int(self.config['NB_sensor_channels'] / 5),
                                             hidden_size=256, batch_first=True, bidirectional=True)
                    elif self.config["NB_sensor_channels"] == 126:
                        self.fc3_N = nn.LSTM(input_size=self.config['num_filters'] * 18,
                                             hidden_size=256, batch_first=True, bidirectional=True)
            elif self.config["aggregate"] == "FC":
                if self.config["network"] == "cnn_imu_tpp":
                    self.fc3_N = nn.Linear(self.tpp.get_output(), 256)
                else:
                    if self.config["reshape_input"]:
                        if self.config["NB_sensor_channels"] == 27:
                            self.fc3_N = nn.Linear(self.config['num_filters'] * int(Wx) *
                                                   int(self.config['NB_sensor_channels'] / 9), 256)
                        elif self.config["NB_sensor_channels"] == 30:
                            self.fc3_N = nn.Linear(self.config['num_filters'] * int(Wx) *
                                                   int(self.config['NB_sensor_channels'] / 15), 256)
                        elif self.config["NB_sensor_channels"] == 126:
                            self.fc3_N = nn.Linear(self.config['num_filters'] * int(Wx) * 6, 256)
                    else:
                        if self.config["NB_sensor_channels"] == 27:
                            self.fc3_N = nn.Linear(self.config['num_filters'] * int(Wx) *
                                                   int(self.config['NB_sensor_channels'] / 3), 256)
                        elif self.config["NB_sensor_channels"] == 30:
                            self.fc3_N = nn.Linear(self.config['num_filters'] * int(Wx) *
                                                   int(self.config['NB_sensor_channels'] / 5), 256)
                        elif self.config["NB_sensor_channels"] == 126:
                            self.fc3_N = nn.Linear(self.config['num_filters'] * int(Wx) * 18, 256)

            # RA
            self.conv_RA_1_1 = nn.Conv2d(in_channels=in_channels,
                                         out_channels=self.config['num_filters'],
                                         kernel_size=(self.config['filter_size'], 1),
                                         stride=1, padding=padding)

            self.conv_RA_1_2 = nn.Conv2d(in_channels=self.config['num_filters'],
                                         out_channels=self.config['num_filters'],
                                         kernel_size=(self.config['filter_size'], 1),
                                         stride=1, padding=padding)

            self.conv_RA_2_1 = nn.Conv2d(in_channels=self.config['num_filters'],
                                         out_channels=self.config['num_filters'],
                                         kernel_size=(self.config['filter_size'], 1),
                                         stride=1, padding=padding)

            self.conv_RA_2_2 = nn.Conv2d(in_channels=self.config['num_filters'],
                                         out_channels=self.config['num_filters'],
                                         kernel_size=(self.config['filter_size'], 1),
                                         stride=1, padding=padding)

            if self.BN_bool:
                self.BN_RA = nn.BatchNorm2d(self.config['num_filters'])

            if self.config["aggregate"] == "FCN":
                self.fc3_RA = nn.Conv2d(in_channels=self.config['num_filters'],
                                        out_channels=256, kernel_size=(1, 1), stride=1, padding=0)
            elif self.config["aggregate"] == "LSTM":
                if self.config["reshape_input"]:
                    if self.config["NB_sensor_channels"] == 27:
                        self.fc3_RA = nn.LSTM(input_size=self.config['num_filters'] *
                                                         int(self.config['NB_sensor_channels'] / 9),
                                              hidden_size=256, batch_first=True, bidirectional=True)
                    elif self.config["NB_sensor_channels"] == 30:
                        self.fc3_RA = nn.LSTM(input_size=self.config['num_filters'] *
                                                         int(self.config['NB_sensor_channels'] / 15),
                                              hidden_size=256, batch_first=True, bidirectional=True)
                    elif self.config["NB_sensor_channels"] == 126:
                        self.fc3_RA = nn.LSTM(input_size=self.config['num_filters'] * 10,
                                              hidden_size=256, batch_first=True, bidirectional=True)
                else:
                    if self.config["NB_sensor_channels"] == 27:
                        self.fc3_RA = nn.LSTM(input_size=self.config['num_filters'] *
                                                         int(self.config['NB_sensor_channels'] / 3),
                                              hidden_size=256, batch_first=True, bidirectional=True)
                    elif self.config["NB_sensor_channels"] == 30:
                        self.fc3_RA = nn.LSTM(input_size=self.config['num_filters'] *
                                                         int(self.config['NB_sensor_channels'] / 5),
                                              hidden_size=256, batch_first=True, bidirectional=True)
                    elif self.config["NB_sensor_channels"] == 126:
                        self.fc3_RA = nn.LSTM(input_size=self.config['num_filters'] * 30,
                                              hidden_size=256, batch_first=True, bidirectional=True)
            elif self.config["aggregate"] == "FC":
                if self.config["network"] == "cnn_imu_tpp":
                    self.fc3_RA = nn.Linear(self.tpp.get_output(), 256)
                else:
                    if self.config["reshape_input"]:
                        if self.config["NB_sensor_channels"] == 27:
                            self.fc3_RA = nn.Linear(self.config['num_filters'] * int(Wx) *
                                                    int(self.config['NB_sensor_channels'] / 9), 256)
                        elif self.config["NB_sensor_channels"] == 30:
                            self.fc3_RA = nn.Linear(self.config['num_filters'] * int(Wx) *
                                                    int(self.config['NB_sensor_channels'] / 15), 256)
                        elif self.config["NB_sensor_channels"] == 126:
                            self.fc3_RA = nn.Linear(self.config['num_filters'] * int(Wx) * 10, 256)
                    else:
                        if self.config["NB_sensor_channels"] == 27:
                            self.fc3_RA = nn.Linear(self.config['num_filters'] * int(Wx) *
                                                    int(self.config['NB_sensor_channels'] / 3), 256)
                        elif self.config["NB_sensor_channels"] == 30:
                            self.fc3_RA = nn.Linear(self.config['num_filters'] * int(Wx) *
                                                    int(self.config['NB_sensor_channels'] / 5), 256)
                        elif self.config["NB_sensor_channels"] == 126:
                            self.fc3_RA = nn.Linear(self.config['num_filters'] * int(Wx) * 30, 256)

            # RL
            self.conv_RL_1_1 = nn.Conv2d(in_channels=in_channels,
                                         out_channels=self.config['num_filters'],
                                         kernel_size=(self.config['filter_size'], 1),
                                         stride=1, padding=padding)

            self.conv_RL_1_2 = nn.Conv2d(in_channels=self.config['num_filters'],
                                         out_channels=self.config['num_filters'],
                                         kernel_size=(self.config['filter_size'], 1),
                                         stride=1, padding=padding)

            self.conv_RL_2_1 = nn.Conv2d(in_channels=self.config['num_filters'],
                                         out_channels=self.config['num_filters'],
                                         kernel_size=(self.config['filter_size'], 1),
                                         stride=1, padding=padding)

            self.conv_RL_2_2 = nn.Conv2d(in_channels=self.config['num_filters'],
                                         out_channels=self.config['num_filters'],
                                         kernel_size=(self.config['filter_size'], 1),
                                         stride=1, padding=padding)
            if self.BN_bool:
                self.BN_RL = nn.BatchNorm2d(self.config['num_filters'])

            if self.config["aggregate"] == "FCN":
                self.fc3_RL = nn.Conv2d(in_channels=self.config['num_filters'],
                                        out_channels=256, kernel_size=(1, 1), stride=1, padding=0)
            elif self.config["aggregate"] == "LSTM":
                if self.config["reshape_input"]:
                    if self.config["NB_sensor_channels"] == 30:
                        self.fc3_RL = nn.LSTM(input_size=self.config['num_filters'] *
                                                         int(self.config['NB_sensor_channels'] / 15),
                                              hidden_size=256, batch_first=True, bidirectional=True)
                    elif self.config["NB_sensor_channels"] == 126:
                        self.fc3_RL = nn.LSTM(input_size=self.config['num_filters'] * 8,
                                              hidden_size=256, batch_first=True, bidirectional=True)
                else:
                    if self.config["NB_sensor_channels"] == 30:
                        self.fc3_RL = nn.LSTM(input_size=self.config['num_filters'] *
                                                         int(self.config['NB_sensor_channels'] / 5),
                                              hidden_size=256, batch_first=True, bidirectional=True)
                    elif self.config["NB_sensor_channels"] == 126:
                        self.fc3_RL = nn.LSTM(input_size=self.config['num_filters'] * 24,
                                              hidden_size=256, batch_first=True, bidirectional=True)
            elif self.config["aggregate"] == "FC":
                if self.config["network"] == "cnn_imu_tpp":
                    self.fc3_RL = nn.Linear(self.tpp.get_output(), 256)
                else:
                    if self.config["reshape_input"]:
                        if self.config["NB_sensor_channels"] == 30:
                            self.fc3_RL = nn.Linear(self.config['num_filters'] * int(Wx) *
                                                    int(self.config['NB_sensor_channels'] / 15), 256)
                        elif self.config["NB_sensor_channels"] == 126:
                            self.fc3_RL = nn.Linear(self.config['num_filters'] * int(Wx) * 8, 256)
                    else:
                        if self.config["NB_sensor_channels"] == 30:
                            self.fc3_RL = nn.Linear(self.config['num_filters'] * int(Wx) *
                                                    int(self.config['NB_sensor_channels'] / 5), 256)
                        elif self.config["NB_sensor_channels"] == 126:
                            self.fc3_RL = nn.Linear(self.config['num_filters'] * int(Wx) * 24, 256)

        # MLP
        if self.config["aggregate"] == "FCN":
            if self.config["network"] == "cnn":
                self.fc4 = nn.Conv2d(in_channels=256,
                                     out_channels=256, kernel_size=(1, 1), stride=1, padding=0)
            elif self.config["network"] == "cnn_imu" and self.config["NB_sensor_channels"] in [30, 126]:
                self.fc4 = nn.Conv2d(in_channels=256,
                                     out_channels=256, kernel_size=(1, 1), stride=1, padding=0)
            elif self.config["network"] == "cnn_imu" and self.config["NB_sensor_channels"] == 27:
                self.fc4 = nn.Conv2d(in_channels=256,
                                     out_channels=256, kernel_size=(1, 1), stride=1, padding=0)
        elif self.config["aggregate"] == "FC":
            if self.config["network"] in ["cnn", "cnn_tpp"]:
                self.fc4 = nn.Linear(256, 256)
            elif self.config["network"] in ["cnn_imu", "cnn_imu_tpp"] and \
                    self.config["NB_sensor_channels"] in [30, 126] and not self.limbs3:
                self.fc4 = nn.Linear(256 * 5, 256)
            elif self.config["network"] in ["cnn_imu", "cnn_imu_tpp"] and self.config["NB_sensor_channels"] == 27:
                self.fc4 = nn.Linear(256 * 3, 256)
            elif self.config["network"] in ["cnn_imu", "cnn_imu_tpp"] and \
                    self.config["NB_sensor_channels"] in [30, 126]:
                self.fc4 = nn.Linear(256 * 3, 256)
        elif self.config["aggregate"] == "LSTM":
            if self.config["network"] in ["cnn"]:
                self.fc4 = nn.LSTM(input_size=256 * 2, hidden_size=256, batch_first=True, bidirectional=True)
            elif self.config["network"] == "cnn_imu" and self.config["NB_sensor_channels"] in [30, 126]:
                if self.limbs3:
                    self.fc4 = nn.LSTM(input_size=256 * 6, hidden_size=256, batch_first=True, bidirectional=True)
                else:
                    self.fc4 = nn.LSTM(input_size=256 * 10, hidden_size=256, batch_first=True, bidirectional=True)
                    # The number of input size is double as one has bidirectional LSTM
            elif self.config["network"] == "cnn_imu" and self.config["NB_sensor_channels"] == 27:
                self.fc4 = nn.LSTM(input_size=256 * 6, hidden_size=256, batch_first=True, bidirectional=True)
                # The number of input size is double as one has bidirectional LSTM

        if self.config["aggregate"] == "FCN":
            if self.config['output'] == 'softmax':
                self.fc5 = nn.Conv2d(in_channels=256,
                                     out_channels=self.config['num_classes'], kernel_size=(1, 1), stride=1, padding=0)
            elif self.config['output'] == 'attribute':
                self.fc5 = nn.Conv2d(in_channels=256,
                                     out_channels=self.config['num_attributes'],
                                     kernel_size=(1, 1), stride=1, padding=0)
            elif self.config['output'] == 'identity':
                self.fc5 = nn.Conv2d(in_channels=256,
                                     out_channels=self.config['num_classes'], kernel_size=(1, 1), stride=1, padding=0)
        elif self.config["aggregate"] == "FC":
            if self.config['output'] == 'softmax':
                self.fc5 = nn.Linear(256, self.config['num_classes'])
            elif self.config['output'] == 'attribute':
                self.fc5 = nn.Linear(256, self.config['num_attributes'])
            elif self.config['output'] == 'identity':
                self.fc5 = nn.Linear(256, self.config['num_classes'])
        elif self.config["aggregate"] == "LSTM":
            if self.config['output'] == 'softmax':
                self.fc5 = nn.Linear(512, self.config['num_classes'])
            elif self.config['output'] == 'attribute':
                self.fc5 = nn.Linear(512, self.config['num_attributes'])
                # The number of input size is double as one has bidirectional LSTM

        if self.config["reshape_input"] or self.limbs3:
            self.avgpool = nn.AvgPool2d(kernel_size=[1, int(self.config['NB_sensor_channels'] / 3)])
        else:
            self.avgpool = nn.AvgPool2d(kernel_size=[1, self.config['NB_sensor_channels']])

        self.softmax = nn.Softmax()

        self.sigmoid = nn.Sigmoid()

        return

    def forward(self, x):
        '''
        Forwards function, required by torch.

        @param x: batch [batch, 1, Channels, Time], Channels = Sensors * 3 Axis
        @return x: Output of the network, either Softmax or Attribute
        '''

        # fft_output, fft = spectral_pooling(x)
        if self.config["reshape_input"]:
            x = x.permute(0, 2, 1, 3)
            x = x.view(x.size()[0], x.size()[1], int(x.size()[3] / 3), 3)
            x = x.permute(0, 3, 1, 2)

        # Selecting the one ot the two networks, tCNN or tCNN-IMU
        if self.config["network"] == "cnn" or self.config["network"] == "cnn_tpp":
            x = self.tcnn(x)
        elif self.config["network"] == "cnn_imu" or self.config["network"] == "cnn_imu_tpp":
            if self.config["dataset"] in ['motionminers_real', 'motionminers_flw', 'motionminers_flw100',
                                          'motionminers_real100']:
                x_LA, x_N, x_RA = self.tcnn_imu(x)
                if self.config["aggregate"] == "FCN":
                    x = torch.cat((x_LA, x_N, x_RA), 3)
                elif self.config["aggregate"] == "FC":
                    x = torch.cat((x_LA, x_N, x_RA), 1)
                elif self.config["aggregate"] == "LSTM":
                    x = torch.cat((x_LA, x_N, x_RA), 2)
            else:
                if self.limbs3:
                    x_LA, x_N, x_RA = self.tcnn_imu(x)
                    if self.config["aggregate"] == "FCN":
                        x = torch.cat((x_LA, x_N, x_RA), 3)
                    elif self.config["aggregate"] == "FC":
                        x = torch.cat((x_LA, x_N, x_RA), 1)
                    elif self.config["aggregate"] == "LSTM":
                        x = torch.cat((x_LA, x_N, x_RA), 2)
                else:
                    x_LA, x_LL, x_N, x_RA, x_RL = self.tcnn_imu(x)
                    if self.config["aggregate"] == "FCN":
                        x = torch.cat((x_LA, x_LL, x_N, x_RA, x_RL), 3)
                    elif self.config["aggregate"] == "FC":
                        x = torch.cat((x_LA, x_LL, x_N, x_RA, x_RL), 1)
                        if self.config["storing_acts"]:
                            self.save_acts(x, "x_concat")
                    elif self.config["aggregate"] == "LSTM":
                        x = torch.cat((x_LA, x_LL, x_N, x_RA, x_RL), 2)

        # Selecting MLP, either FC or FCN
        if self.config["aggregate"] == "FCN":
            # x = F.dropout(x, training=self.training)
            x = F.dropout2d(x, training=self.training)
            x = F.relu(self.fc4(x))
            # x = F.dropout(x, training=self.training)
            x = F.dropout2d(x, training=self.training)
            x = self.fc5(x)
            # x = F.relu(self.fc5(x))
            x = self.avgpool(x)
            x = x.view(x.size()[0], x.size()[1], x.size()[2])
            x = x.permute(0, 2, 1)
        elif self.config["aggregate"] == "FC":
            x = F.dropout(x, training=self.training)
            x = F.relu(self.fc4(x))
            if self.config["storing_acts"]:
                self.save_acts(x, "x_fc4")
            x = F.dropout(x, training=self.training)
            x = self.fc5(x)
            if self.config["storing_acts"]:
                self.save_acts(x, "x_fc5")
        elif self.config["aggregate"] == "LSTM":
            x = F.dropout(x, training=self.training)
            x = F.relu(self.fc4(x)[0])
            x = F.dropout(x, training=self.training)
            # x = x[:, -1]
            x = self.fc5(x)

        if self.training:
            if self.config['output'] == 'attribute':
                x = self.sigmoid(x)
        elif not self.training:
            if self.config['output'] == 'softmax' or self.config['output'] == 'identity':
                if self.config["aggregate"] == "FCN":
                    x = x.reshape(-1, x.size()[2])
                if self.config["aggregate"] == "LSTM":
                    x = x.reshape(-1, x.size()[2])
                x = self.softmax(x)
            elif self.config['output'] == 'attribute':
                if self.config["aggregate"] == "FCN":
                    x = x.reshape(-1, x.size()[2])
                if self.config["aggregate"] == "LSTM":
                    x = x.reshape(-1, x.size()[2])
                x = self.sigmoid(x)
                if self.config["storing_acts"]:
                    self.save_acts(x, "x_attrs")

        return x
        # return x11.clone(), x12.clone(), x21.clone(), x22.clone(), x

    def init_weights(self):
        '''
        Applying initialisation of layers
        '''
        self.apply(Network._init_weights_orthonormal)
        return

    @staticmethod
    def _init_weights_orthonormal(m):
        '''
        Orthonormal Initialissation of layer

        @param m: layer m
        '''
        if isinstance(m, nn.Conv2d):
            # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            # m.weight.data.normal_(0, (2. / n) ** (1 / 2.0))
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias.data, 0)
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias.data, 0)

        return

    def size_feature_map(self, Wx, Hx, F, P, S, type_layer='conv'):
        '''
        Computing size of feature map after convolution or pooling

        @param Wx: Width input
        @param Hx: Height input
        @param F: Filter size
        @param P: Padding
        @param S: Stride
        @param type_layer: conv or pool
        @return Wy: Width output
        @return Hy: Height output
        '''

        if self.config["pooling"] in [3, 4] and self.config["dataset"] in ['mbientlab_quarter', 'gesture',
                                                                           'mocap_quarter', 'virtual_quarter']:
            Pw = P[0]
            Ph = P[1]
        elif self.config["aggregate"] in ["FCN", "LSTM"]:
            Pw = P[0]
            Ph = P[1]
        elif self.config["aggregate"] == "FC":
            Pw = P
            Ph = P

        if type_layer == 'conv':
            Wy = 1 + (Wx - F[0] + 2 * Pw) / S[0]
            Hy = 1 + (Hx - F[1] + 2 * Ph) / S[1]

        elif type_layer == 'pool':
            Wy = 1 + (Wx - F[0] + 2 * Pw) / S[0]
            Hy = 1 + (Hx - F[1] + 2 * Ph) / S[1]

        return Wy, Hy

    def tcnn(self, x):
        '''
        tCNN network

        @param x: input sequence
        @return x: Prediction of the network
        '''
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        if self.config["pooling"] in [1, 2]:
            x = self.spectral_pooling(x, pooling_number=0)
        elif self.config["pooling"] in [3, 4]:
            if self.config["aggregate"] in ["FCN", "LSTM"] or self.config["dataset"] in ['mbientlab_quarter', 'gesture',
                                                                                         'mocap_quarter',
                                                                                         'virtual_quarter']:
                x = F.max_pool2d(x, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
            else:
                x = F.max_pool2d(x, (2, 1))
        # fft_output = x.clone().detach()

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))

        if self.config["pooling"] == 2:
            x = self.spectral_pooling(x, pooling_number=1)
        elif self.config["pooling"] == 4:
            if self.config["aggregate"] in ["FCN", "LSTM"] or self.config["dataset"] in ['mbientlab_quarter', 'gesture',
                                                                                         'mocap_quarter',
                                                                                         'virtual_quarter']:
                x = F.max_pool2d(x, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), ceil_mode=True)
            else:
                x = F.max_pool2d(x, (2, 1))
        # fft_output = x.clone().detach()

        if self.config["network"] == "cnn_tpp":
            if self.BN_bool:
                x = self.BN(x)
            x = self.tpp.tpp(x)
            x = F.relu(self.fc3(x))
        else:
            if self.config["aggregate"] == "FCN":
                x = F.relu(self.fc3(x))
            elif self.config["aggregate"] == "FC":
                # view is reshape
                x = x.reshape((-1, x.size()[1] * x.size()[2] * x.size()[3]))
                x = F.relu(self.fc3(x))
            elif self.config["aggregate"] == "LSTM":
                x = x.permute(0, 2, 1, 3)
                x = x.reshape((x.size()[0], x.size()[1], x.size()[2] * x.size()[3]))
                x = F.relu(self.fc3(x)[0])
        return x

    def tcnn_imu(self, x):
        '''
        tCNN-IMU network
        The parameters will adapt according to the dataset, reshape and output type

        x_LA, x_LL, x_N, x_RA, x_RL

        @param x: input sequence
        @return x_LA: Features from left arm
        @return x_LL: Features from left leg
        @return x_N: Features from Neck or Torso
        @return x_RA: Features from Right Arm
        @return x_RL: Features from Right Leg
        '''
        # LA
        if self.config["storing_acts"]:
            self.save_acts(x, "input")

        if self.config["reshape_input"]:
            if self.config["NB_sensor_channels"] == 27:
                x_LA = F.relu(self.conv_LA_1_1(x[:, :, :, 0:3]))
            if self.config["NB_sensor_channels"] == 30:
                x_LA = F.relu(self.conv_LA_1_1(x[:, :, :, 0:2]))
            elif self.config["NB_sensor_channels"] == 126:
                idx_LA = np.arange(4, 8)
                idx_LA = np.concatenate([idx_LA, np.arange(12, 14)])
                idx_LA = np.concatenate([idx_LA, np.arange(18, 22)])
                x_LA = F.relu(self.conv_LA_1_1(x[:, :, :, idx_LA]))
        else:
            if self.config["NB_sensor_channels"] == 27:
                x_LA = F.relu(self.conv_LA_1_1(x[:, :, :, 0:9]))
            if self.config["NB_sensor_channels"] == 30:
                x_LA = F.relu(self.conv_LA_1_1(x[:, :, :, 0:6]))
            elif self.config["NB_sensor_channels"] == 126:
                idx_LA = np.arange(12, 24)
                idx_LA = np.concatenate([idx_LA, np.arange(36, 42)])
                idx_LA = np.concatenate([idx_LA, np.arange(54, 66)])
                x_LA = F.relu(self.conv_LA_1_1(x[:, :, :, idx_LA]))

        if self.config["storing_acts"]:
            self.save_acts(x_LA, "x_LA_1_1")

        x_LA = F.relu(self.conv_LA_1_2(x_LA))
        if self.config["storing_acts"]:
            self.save_acts(x_LA, "x_LA_1_2")

        if self.config["pooling"] in [1, 2]:
            x_LA = self.spectral_pooling(x_LA, pooling_number=0)
        elif self.config["pooling"] in [3, 4]:
            if self.config["aggregate"] in ["FCN", "LSTM"] or self.config["dataset"] in ['mbientlab_quarter', 'gesture',
                                                                                         'mocap_quarter',
                                                                                         'virtual_quarter']:
                x_LA = F.max_pool2d(x_LA, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
            else:
                x_LA = F.max_pool2d(x_LA, (2, 1))

        if self.config["storing_acts"]:
            self.save_acts(x_LA, "x_LA_1_2_pool")

        x_LA = F.relu(self.conv_LA_2_1(x_LA))
        if self.config["storing_acts"]:
            self.save_acts(x_LA, "x_LA_2_1")
        x_LA = F.relu(self.conv_LA_2_2(x_LA))
        if self.config["storing_acts"]:
            self.save_acts(x_LA, "x_LA_2_2")
        if self.config["pooling"] == 2:
            x_LA = self.spectral_pooling(x_LA, pooling_number=1)
        elif self.config["pooling"] == 4:
            if self.config["aggregate"] in ["FCN", "LSTM"] or self.config["dataset"] in ['mbientlab_quarter', 'gesture',
                                                                                         'mocap_quarter',
                                                                                         'virtual_quarter']:
                x_LA = F.max_pool2d(x_LA, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
            else:
                x_LA = F.max_pool2d(x_LA, (2, 1))

        if self.BN_bool:
            x_LA = self.BN_LA(x_LA)
        if self.config["storing_acts"]:
            self.save_acts(x_LA, "x_LA_2_2_pool")

        if self.config["network"] == "cnn_imu_tpp":
            x_LA = self.tpp.tpp(x_LA)
            if self.config["storing_acts"]:
                self.save_acts(x_LA, "x_LA_tpp")
            x_LA = F.relu(self.fc3_LA(x_LA))
            if self.config["storing_acts"]:
                self.save_acts(x_LA, "x_LA_fc3")
        else:
            if self.config["aggregate"] == "FCN":
                x_LA = F.relu(self.fc3_LA(x_LA))
            elif self.config["aggregate"] == "FC":
                # view is reshape
                x_LA = x_LA.reshape(-1, x_LA.size()[1] * x_LA.size()[2] * x_LA.size()[3])
                x_LA = F.relu(self.fc3_LA(x_LA))
            elif self.config["aggregate"] == "LSTM":
                x_LA = x_LA.permute(0, 2, 1, 3)
                x_LA = x_LA.reshape((x_LA.size()[0], x_LA.size()[1], x_LA.size()[2] * x_LA.size()[3]))
                x_LA = F.relu(self.fc3_LA(x_LA)[0])

        # LL
        if self.config["NB_sensor_channels"] in [30, 126]:
            if self.config["reshape_input"]:
                if self.config["NB_sensor_channels"] == 30:
                    x_LL = F.relu(self.conv_LL_1_1(x[:, :, :, 2:4]))
                elif self.config["NB_sensor_channels"] == 126:
                    idx_LL = np.arange(8, 12)
                    idx_LL = np.concatenate([idx_LL, np.arange(14, 18)])
                    x_LL = F.relu(self.conv_LA_1_1(x[:, :, :, idx_LL]))
            else:
                if self.config["NB_sensor_channels"] == 30:
                    x_LL = F.relu(self.conv_LL_1_1(x[:, :, :, 6:12]))
                elif self.config["NB_sensor_channels"] == 126:
                    idx_LL = np.arange(24, 36)
                    idx_LL = np.concatenate([idx_LL, np.arange(42, 54)])
                    x_LL = F.relu(self.conv_LA_1_1(x[:, :, :, idx_LL]))

            x_LL = F.relu(self.conv_LL_1_2(x_LL))
            if self.config["pooling"] in [1, 2]:
                x_LL = self.spectral_pooling(x_LL, pooling_number=0)
            elif self.config["pooling"] in [3, 4]:
                if self.config["aggregate"] in ["FCN", "LSTM"] or self.config["dataset"] in ['mbientlab_quarter',
                                                                                             'gesture', 'mocap_quarter',
                                                                                             'virtual_quarter']:
                    x_LL = F.max_pool2d(x_LL, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
                else:
                    x_LL = F.max_pool2d(x_LL, (2, 1))
            x_LL = F.relu(self.conv_LL_2_1(x_LL))
            x_LL = F.relu(self.conv_LL_2_2(x_LL))
            if self.config["pooling"] == 2:
                x_LL = self.spectral_pooling(x_LL, pooling_number=1)
            elif self.config["pooling"] == 4:
                if self.config["aggregate"] in ["FCN", "LSTM"] or self.config["dataset"] in ['mbientlab_quarter',
                                                                                             'gesture', 'mocap_quarter',
                                                                                             'virtual_quarter']:
                    x_LL = F.max_pool2d(x_LL, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
                else:
                    x_LL = F.max_pool2d(x_LL, (2, 1))

            if self.BN_bool:
                x_LL = self.BN_LL(x_LL)
            if self.config["network"] == "cnn_imu_tpp":
                x_LL = self.tpp.tpp(x_LL)
                x_LL = F.relu(self.fc3_LL(x_LL))
            else:
                if self.config["aggregate"] == "FCN":
                    x_LL = F.relu(self.fc3_LL(x_LL))
                elif self.config["aggregate"] == "FC":
                    # view is reshape
                    x_LL = x_LL.reshape(-1, x_LL.size()[1] * x_LL.size()[2] * x_LL.size()[3])
                    x_LL = F.relu(self.fc3_LL(x_LL))
                elif self.config["aggregate"] == "LSTM":
                    x_LL = x_LL.permute(0, 2, 1, 3)
                    x_LL = x_LL.reshape((x_LL.size()[0], x_LL.size()[1], x_LL.size()[2] * x_LL.size()[3]))
                    x_LL = F.relu(self.fc3_LL(x_LL)[0])

        # N
        if self.config["reshape_input"]:
            if self.config["NB_sensor_channels"] == 27:
                x_N = F.relu(self.conv_N_1_1(x[:, :, :, 3:6]))
            if self.config["NB_sensor_channels"] == 30:
                x_N = F.relu(self.conv_N_1_1(x[:, :, :, 4:6]))
            elif self.config["NB_sensor_channels"] == 126:
                idx_N = np.arange(0, 4)
                idx_N = np.concatenate([idx_N, np.arange(40, 42)])
                x_N = F.relu(self.conv_LA_1_1(x[:, :, :, idx_N]))
        else:
            if self.config["NB_sensor_channels"] == 27:
                x_N = F.relu(self.conv_N_1_1(x[:, :, :, 9:18]))
            if self.config["NB_sensor_channels"] == 30:
                x_N = F.relu(self.conv_N_1_1(x[:, :, :, 12:18]))
            elif self.config["NB_sensor_channels"] == 126:
                idx_N = np.arange(0, 12)
                idx_N = np.concatenate([idx_N, np.arange(120, 126)])
                x_N = F.relu(self.conv_LA_1_1(x[:, :, :, idx_N]))
        x_N = F.relu(self.conv_N_1_2(x_N))
        if self.config["pooling"] == 1 or self.config["pooling"] == 2:
            x_N = self.spectral_pooling(x_N, pooling_number=0)
        elif self.config["pooling"] in [3, 4]:
            if self.config["aggregate"] in ["FCN", "LSTM"] or self.config["dataset"] in ['mbientlab_quarter', 'gesture',
                                                                                         'mocap_quarter',
                                                                                         'virtual_quarter']:
                x_N = F.max_pool2d(x_N, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
            else:
                x_N = F.max_pool2d(x_N, (2, 1))
        x_N = F.relu(self.conv_N_2_1(x_N))
        x_N = F.relu(self.conv_N_2_2(x_N))
        if self.config["pooling"] == 2:
            x_N = self.spectral_pooling(x_N, pooling_number=1)
        elif self.config["pooling"] == 4:
            if self.config["aggregate"] in ["FCN", "LSTM"] or self.config["dataset"] in ['mbientlab_quarter', 'gesture',
                                                                                         'mocap_quarter',
                                                                                         'virtual_quarter']:
                x_N = F.max_pool2d(x_N, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
            else:
                x_N = F.max_pool2d(x_N, (2, 1))

        if self.BN_bool:
            x_N = self.BN_N(x_N)

        if self.config["network"] == "cnn_imu_tpp":
            x_N = self.tpp.tpp(x_N)
            x_N = F.relu(self.fc3_N(x_N))
        else:
            if self.config["aggregate"] == "FCN":
                x_N = F.relu(self.fc3_N(x_N))
            elif self.config["aggregate"] == "FC":
                # view is reshape
                x_N = x_N.reshape(-1, x_N.size()[1] * x_N.size()[2] * x_N.size()[3])
                x_N = F.relu(self.fc3_N(x_N))
            elif self.config["aggregate"] == "LSTM":
                x_N = x_N.permute(0, 2, 1, 3)
                x_N = x_N.reshape((x_N.size()[0], x_N.size()[1], x_N.size()[2] * x_N.size()[3]))
                x_N = F.relu(self.fc3_N(x_N)[0])

        # RA
        if self.config["reshape_input"]:
            if self.config["NB_sensor_channels"] == 27:
                x_RA = F.relu(self.conv_RA_1_1(x[:, :, :, 6:9]))
            if self.config["NB_sensor_channels"] == 30:
                x_RA = F.relu(self.conv_RA_1_1(x[:, :, :, 6:8]))
            elif self.config["NB_sensor_channels"] == 126:
                idx_RA = np.arange(22, 26)
                idx_RA = np.concatenate([idx_RA, np.arange(30, 32)])
                idx_RA = np.concatenate([idx_RA, np.arange(36, 40)])
                x_RA = F.relu(self.conv_LA_1_1(x[:, :, :, idx_RA]))
        else:
            if self.config["NB_sensor_channels"] == 27:
                x_RA = F.relu(self.conv_RA_1_1(x[:, :, :, 18:27]))
            if self.config["NB_sensor_channels"] == 30:
                x_RA = F.relu(self.conv_RA_1_1(x[:, :, :, 18:24]))
            elif self.config["NB_sensor_channels"] == 126:
                idx_RA = np.arange(66, 78)
                idx_RA = np.concatenate([idx_RA, np.arange(90, 96)])
                idx_RA = np.concatenate([idx_RA, np.arange(108, 120)])
                x_RA = F.relu(self.conv_LA_1_1(x[:, :, :, idx_RA]))

        x_RA = F.relu(self.conv_RA_1_2(x_RA))
        if self.config["pooling"] in [1, 2]:
            x_RA = self.spectral_pooling(x_RA, pooling_number=0)
        elif self.config["pooling"] in [3, 4]:
            if self.config["aggregate"] in ["FCN", "LSTM"] or self.config["dataset"] in ['mbientlab_quarter', 'gesture',
                                                                                         'mocap_quarter',
                                                                                         'virtual_quarter']:
                x_RA = F.max_pool2d(x_RA, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
            else:
                x_RA = F.max_pool2d(x_RA, (2, 1))
        x_RA = F.relu(self.conv_RA_2_1(x_RA))
        x_RA = F.relu(self.conv_RA_2_2(x_RA))
        if self.config["pooling"] == 2:
            x_RA = self.spectral_pooling(x_RA, pooling_number=1)
        elif self.config["pooling"] == 4:
            if self.config["aggregate"] in ["FCN", "LSTM"] or self.config["dataset"] in ['mbientlab_quarter', 'gesture',
                                                                                         'mocap_quarter',
                                                                                         'virtual_quarter']:
                x_RA = F.max_pool2d(x_RA, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
            else:
                x_RA = F.max_pool2d(x_RA, (2, 1))

        if self.BN_bool:
            x_RA = self.BN_RA(x_RA)
        if self.config["network"] == "cnn_imu_tpp":
            x_RA = self.tpp.tpp(x_RA)
            x_RA = F.relu(self.fc3_RA(x_RA))
        else:
            if self.config["aggregate"] == "FCN":
                x_RA = F.relu(self.fc3_RA(x_RA))
            elif self.config["aggregate"] == "FC":
                # view is reshape
                x_RA = x_RA.reshape(-1, x_RA.size()[1] * x_RA.size()[2] * x_RA.size()[3])
                x_RA = F.relu(self.fc3_RA(x_RA))
            elif self.config["aggregate"] == "LSTM":
                x_RA = x_RA.permute(0, 2, 1, 3)
                x_RA = x_RA.reshape((x_RA.size()[0], x_RA.size()[1], x_RA.size()[2] * x_RA.size()[3]))
                x_RA = F.relu(self.fc3_RA(x_RA)[0])

        # RL
        if self.config["NB_sensor_channels"] in [30, 126]:
            if self.config["reshape_input"]:
                if self.config["NB_sensor_channels"] == 30:
                    x_RL = F.relu(self.conv_RL_1_1(x[:, :, :, 8:10]))
                elif self.config["NB_sensor_channels"] == 126:
                    idx_RL = np.arange(26, 30)
                    idx_RL = np.concatenate([idx_RL, np.arange(32, 36)])
                    x_RL = F.relu(self.conv_LA_1_1(x[:, :, :, idx_RL]))
            else:
                if self.config["NB_sensor_channels"] == 30:
                    x_RL = F.relu(self.conv_RL_1_1(x[:, :, :, 24:30]))
                elif self.config["NB_sensor_channels"] == 126:
                    idx_RL = np.arange(78, 90)
                    idx_RL = np.concatenate([idx_RL, np.arange(96, 108)])
                    x_RL = F.relu(self.conv_LA_1_1(x[:, :, :, idx_RL]))

            x_RL = F.relu(self.conv_RL_1_2(x_RL))
            if self.config["pooling"] in [1, 2]:
                x_RL = self.spectral_pooling(x_RL, pooling_number=0)
            elif self.config["pooling"] in [3, 4]:
                if self.config["aggregate"] in ["FCN", "LSTM"] or self.config["dataset"] in ['mbientlab_quarter',
                                                                                             'gesture', 'mocap_quarter',
                                                                                             'virtual_quarter']:
                    x_RL = F.max_pool2d(x_RL, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
                else:
                    x_RL = F.max_pool2d(x_RL, (2, 1))
            x_RL = F.relu(self.conv_RL_2_1(x_RL))
            x_RL = F.relu(self.conv_RL_2_2(x_RL))
            if self.config["pooling"] == 2:
                x_RL = self.spectral_pooling(x_RL, pooling_number=1)
            elif self.config["pooling"] == 4:
                if self.config["aggregate"] in ["FCN", "LSTM"] or self.config["dataset"] in ['mbientlab_quarter',
                                                                                             'gesture', 'mocap_quarter',
                                                                                             'virtual_quarter']:
                    x_RL = F.max_pool2d(x_RL, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
                else:
                    x_RL = F.max_pool2d(x_RL, (2, 1))

            if self.BN_bool:
                x_RL = self.BN_(x_RL)
            if self.config["network"] == "cnn_imu_tpp":
                x_RL = self.tpp.tpp(x_RL)
                x_RL = F.relu(self.fc3_RL(x_RL))
            else:
                if self.config["aggregate"] == "FCN":
                    x_RL = F.relu(self.fc3_RL(x_RL))
                elif self.config["aggregate"] == "FC":
                    # view is reshape
                    x_RL = x_RL.reshape(-1, x_RL.size()[1] * x_RL.size()[2] * x_RL.size()[3])
                    x_RL = F.relu(self.fc3_RL(x_RL))
                elif self.config["aggregate"] == "LSTM":
                    x_RL = x_RL.permute(0, 2, 1, 3)
                    x_RL = x_RL.reshape((x_RL.size()[0], x_RL.size()[1], x_RL.size()[2] * x_RL.size()[3]))
                    x_RL = F.relu(self.fc3_RL(x_RL)[0])

        if self.config["NB_sensor_channels"] == 27 or self.limbs3:
            return x_LA, x_N, x_RA
        else:
            return x_LA, x_LL, x_N, x_RA, x_RL

    def spectral_pooling(self, x, pooling_number=0):
        '''
        Carry out a spectral pooling.
        torch.rfft(x, signal_ndim, normalized, onesided)
        signal_ndim takes into account the signal_ndim dimensions stranting from the last one
        onesided if True, outputs only the positives frequencies, under the nyquist frequency

        @param x: input sequence
        @return x: output of spectral pooling
        '''
        # xpool = F.max_pool2d(x, (2, 1))

        x = x.permute(0, 1, 3, 2)

        # plt.figure()
        # f, axarr = plt.subplots(5, 1)

        # x_plt = x[0, 0].to("cpu", torch.double).detach()
        # axarr[0].plot(x_plt[0], current_class_label='input')

        # fft = torch.rfft(x, signal_ndim=1, normalized=True, onesided=True)
        fft = torch.fft.rfft(x, norm="forward")
        if self.config["storing_acts"]:
            self.save_acts(fft, "x_LA_fft")
        # fft2 = torch.rfft(x, signal_ndim=1, normalized=False, onesided=False)

        # fft_plt = fft[0, 0].to("cpu", torch.double).detach()
        # fft_plt = torch.norm(fft_plt, dim=2)
        # axarr[1].plot(fft_plt[0], 'o', current_class_label='fft')

        # x = fft[:, :, :, :int(fft.shape[3] / 2)]
        x = fft[:, :, :, :int(math.ceil(fft.shape[3] / 2))]
        if self.config["storing_acts"]:
            self.save_acts(x, "x_LA_fft_2")

        # fftx_plt = x[0, 0].to("cpu", torch.double).detach()
        # fftx_plt = torch.norm(fftx_plt, dim=2)
        # axarr[2].plot(fftx_plt[0], 'o', current_class_label='fft')

        # x = torch.irfft(x, signal_ndim=1, normalized=True, onesided=True)
        x = torch.fft.irfft(x, norm="forward")
        if self.config["storing_acts"]:
            self.save_acts(x, "x_LA_ifft")

        x = x[:, :, :, :self.pooling_Wx[pooling_number]]
        if self.config["storing_acts"]:
            self.save_acts(x, "x_LA_ifft_pool")

        # x_plt = x[0, 0].to("cpu", torch.double).detach()
        # axarr[3].plot(x_plt[0], current_class_label='input')

        x = x.permute(0, 1, 3, 2)

        # fft2_plt = fft2[0, 0].to("cpu", torch.double).detach()
        # fft2_plt = torch.norm(fft2_plt, dim=2)
        # print(fft2_plt.size(), 'max: {}'.format(torch.max(fft2_plt)), 'min: {}'.format(torch.min(fft2_plt)))
        # axarr[4].plot(fft2_plt[0], 'o', current_class_label='fft')

        # xpool = xpool.permute(0, 1, 3, 2)
        # x_plt = xpool[0, 0].to("cpu", torch.double).detach()
        # axarr[3].plot(x_plt[0], current_class_label='input')

        # plt.waitforbuttonpress(0)
        # plt.close()

        return x

    def save_acts(self, x, name_file):
        # Storing the sequences
        obj = {"input": x.to("cpu", torch.double).numpy()}
        f = open(self.config['folder_exp'] + "activations/" + name_file + ".pkl", 'wb')
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

        return


def load_network(path) -> Network:
    """"""
    state_dict = torch.load(path)
    # print(state_dict.keys())
    net_config = state_dict["network_config"]

    if "dataset" not in net_config.keys():
        net_config["dataset"] = f"mbientlab{window_size}"
    if "pooling" not in net_config.keys():
        net_config["pooling"] = 0
    if "aggregate" not in net_config.keys():
        net_config['aggregate']= 'FC'
    if "storing_acts" not in net_config.keys():
        net_config["storing_acts"] = False

    # net = Network(state_dict["network_config"]) # incomplete config. gives keyerrors
    #net = Network(config)
    net = Network(net_config)

    print(net.config['sliding_window_length'])
    net.load_state_dict(state_dict["state_dict"])
    net.eval()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    net.to(device=device, dtype=torch.float)

    # print(net)
    return net




mean_values = np.array([-0.59011078, 0.21370987, 0.35121016, 0.92213551, 0.16381447,
                        -1.4104172, 0.0227385, 1.01524423, -0.1326606, 1.50302222,
                        1.6741917, -1.36223996, -0.96851859, -0.07210883, 0.13965264,
                        -0.00339991, 1.10663298, 0.48543698, -0.60798819, -0.23433834,
                        0.41563179, -1.09679218, 0.70833077, 0.37622293, -0.2071909,
                        -0.87027332, -0.20594663, -0.13839382, 0.1675007, 0.70815734])
mean_values = np.reshape(mean_values, [1, 30])
std_values = np.array([1.02790431, 0.54293909, 0.60919268, 56.30538052, 72.91016666,
                       78.53805221, 0.89515912, 0.54585097, 0.62408347, 75.26524784,
                       91.73655101, 60.16483688, 0.33170985, 0.21570707, 0.29992192,
                       42.02785466, 23.25204603, 18.41281644, 0.43064442, 0.51308243,
                       0.45229039, 72.98513239, 73.16297869, 72.16814269, 0.79006157,
                       0.45560435, 0.5095047, 35.01586674, 58.97545938, 61.68184099])
std_values = np.reshape(std_values, [1, 30])

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

mean_values = torch.from_numpy(np.repeat(mean_values, window_size, axis=0).T[None, None, :, :]).to(device=device,
                                                                                                   dtype=torch.float)
std_values = torch.from_numpy(np.repeat(std_values, window_size, axis=0).T[None, None, :, :]).to(device=device,
                                                                                                 dtype=torch.float)

max_values = mean_values + 2 * std_values
min_values = mean_values - 2 * std_values


def norm_mbientlab(data):
    data_norm = (data - min_values) / (max_values - min_values)
    data_norm[data_norm > 1] = 1
    data_norm[data_norm < 0] = 0
    return data_norm


if __name__ == "__main__":
    print("loading network")
    net = load_network("../network.pt")
    print("network loaded")
