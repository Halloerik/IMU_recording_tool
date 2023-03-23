'''
Created on 11.06.2020

@author: Erik
'''
import math

import numpy as np
# from mbientlab.metawear.cbindings import GyroBoschOdr,GyroBoschRange
from mbientlab.metawear.metawear import GyroBoschOdr, GyroBoschRange, AccBmi160Odr, AccBoschRange

settings = {
    't_adress': 'CB:84:80:F9:18:41',
    'la_adress': 'D4:6A:6A:EF:55:23',
    'ra_adress': 'EE:2D:7D:2E:3A:72',
    'll_adress': 'C8:73:59:6B:FB:D1',
    'rl_adress': 'FF:EA:47:38:42:8E',
    'folder': 'recordings',
    'recording_interval': 140*1000,  # 140 seconds

    # Sensor parameters see:
    # https://mbientlab.com/documents/metawear/cpp/latest/accelerometer_8h.html
    # https://mbientlab.com/documents/metawear/cpp/latest/gyro__bmi160_8h.html
    # https://mbientlab.com/documents/metawear/cpp/latest/gyro__bosch_8h.html
    # 'acc_odr': 200.0,
    # 'acc_range': 8.0,
    # 'gyro_odr': 8.0,  # because of the MblMwGyroBmi169Odr
    # 'gyro_range': 1.0,
    'acc_odr': AccBmi160Odr._100Hz,
    'acc_range': AccBoschRange._8G,
    'gyro_odr': GyroBoschOdr._100Hz,
    'gyro_range': GyroBoschRange._250dps,

    # Connection parameters see:
    # https://mbientlab.com/documents/metawear/cpp/latest/settings_8h.html#a1cf3cae052fe7981c26124340a41d66d
    'min_conn_interval': 7.5,
    'max_conn_interval': 7.5,
    'latency': 0,
    'timeout': 6000,

    # Other timers
    'graph_interval': 1000,  # Milliseconds
    'analysis_interval': 1000,  # Milliseconds
}

state = {'recording': False,
         'end': False,
         'final_folder': '',
         't_connected': False,
         'la_connected': False,
         'ra_connected': False,
         'll_connected': False,
         'rl_connected': False, }


# if not os.path.exists(settings["folder"]):
#    os.makedirs(settings["folder"])

t_data =  np.zeros((200, 6))
la_data = np.zeros((200, 6))
ra_data = np.zeros((200, 6))
ll_data = np.zeros((200, 6))
rl_data = np.zeros((200, 6))

t_energy =  np.zeros((200,))
la_energy = np.zeros((200,))
ra_energy = np.zeros((200,))
ll_energy = np.zeros((200,))
rl_energy = np.zeros((200,))
