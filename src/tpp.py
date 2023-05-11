'''
Created on March 10, 2021

@author: fmoya
fernando.moya@tu-dortmund.de
'''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging


class TPP(nn.Module):
    '''
    classdocs
    '''

    def __init__(self, config, levels=[1, 2, 4, 8, 16], mode="max"):
        '''
        Constructor.
        '''

        super(TPP, self).__init__()

        logging.info('            Metrics: Constructor')
        self.config = config
        self.levels = levels
        self.mode = mode

        return

    def forward(self, x):
        return self.tpp(x)

    def get_output(self):

        out = 0
        for level in self.levels:
            out += self.config["num_filters"] * level
        return out

    def tpp(self, x):

        num_sample = x.size(0)

        # [T, C]
        spacial_dim = [int(math.ceil(x.size(2))), int(math.ceil(x.size(3)))]
        tpp_pooled = []
        for i, level in enumerate(self.levels):
            w_kernel = spacial_dim[1]
            h_kernel = int(math.ceil(spacial_dim[0] / level))
            h_pad1 = int(math.floor((h_kernel * level - spacial_dim[0]) / 2))
            h_pad2 = int(math.ceil((h_kernel * level - spacial_dim[0]) / 2))
            assert h_pad1 + h_pad2 == (h_kernel * level - spacial_dim[0])

            padded_input = F.pad(input=x, pad=(0, 0, h_pad1, h_pad2), mode='constant', value=0)
            if self.mode == "max":
                pool = nn.MaxPool2d((h_kernel, w_kernel), stride=(h_kernel, w_kernel), padding=(0, 0))
            elif self.mode == "avg":
                pool = nn.AvgPool2d((h_kernel, w_kernel), stride=(h_kernel, w_kernel), padding=(0, 0))

            tpp_pooled.append(pool(padded_input))
            #padded_input = pool(padded_input)

            if i == 0:
                tpp_out = tpp_pooled[i].reshape(num_sample, -1)
            else:
                tpp_out = torch.cat((tpp_out, tpp_pooled[i].reshape(num_sample, -1)), 1)
                tpp_pooled[i].reshape(num_sample, -1)
        return tpp_out
