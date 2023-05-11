'''
Created on 11.06.2020

@author: Erik Altermann, Fernando Moya Rueda, Arthur Matei
@email: erik.altermann@tu-dortmund.de, 	fernando.moya@tu-dortmund.de, arthur.matei@tu-dortmund.de
'''
import math

import numpy as np
# from mbientlab.metawear.cbindings import GyroBoschOdr,GyroBoschRange
from mbientlab.metawear.metawear import GyroBoschOdr, GyroBoschRange, AccBmi160Odr, AccBoschRange

settings = {
    't_address': 'CB:84:80:F9:18:41',
    'la_address': 'D4:6A:6A:EF:55:23',
    'ra_address': 'EE:2D:7D:2E:3A:72',
    'll_address': 'C8:73:59:6B:FB:D1',
    'rl_address': 'FF:EA:47:38:42:8E',
    'folder': 'recordings',
    'recording_interval': 140*1000,  # 140 seconds

    # Sensor parameters see:
    # https://mbientlab.com/documents/metawear/cpp/latest/accelerometer_8h.html
    # https://mbientlab.com/documents/metawear/cpp/latest/gyro__bmi160_8h.html
    # https://mbientlab.com/documents/metawear/cpp/latest/gyro__bosch_8h.html
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

    'connection_retries':10,  # How often the program will try to connect to the sensors before giving up.

    # Other timers
    'graph_interval': 10,  # Milliseconds
    'analysis_interval': 1000,  # Milliseconds
}

# Controls for the sensor threads.
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

# Raw data from each sensor.
t_data =  np.zeros((200, 6))
la_data = np.zeros((200, 6))
ra_data = np.zeros((200, 6))
ll_data = np.zeros((200, 6))
rl_data = np.zeros((200, 6))

# L2 Norm of each sensors raw data.
t_energy =  np.zeros((200,))
la_energy = np.zeros((200,))
ra_energy = np.zeros((200,))
ll_energy = np.zeros((200,))
rl_energy = np.zeros((200,))

