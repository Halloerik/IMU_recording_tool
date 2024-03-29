'''
Created on 11.06.2020

@author: Erik Altermann, Fernando Moya Rueda, Arthur Matei
@email: erik.altermann@tu-dortmund.de, 	fernando.moya@tu-dortmund.de, arthur.matei@tu-dortmund.de
'''
import math
import os
import numpy as np
# from mbientlab.metawear.cbindings import GyroBoschOdr,GyroBoschRange
from mbientlab.metawear.metawear import GyroBoschOdr, GyroBoschRange, AccBmi160Odr, AccBoschRange

settings = {
    't_address': 'CB:84:80:F9:18:41',
    'la_address': 'D4:6A:6A:EF:55:23',
    'ra_address': 'EE:2D:7D:2E:3A:72',
    'll_address': 'C8:73:59:6B:FB:D1',
    'rl_address': 'FF:EA:47:38:42:8E',
    'folder': f'..{os.sep}recordings',
    'recording_interval': 140 * 1000,  # 140 seconds

    # Sensor parameters see:
    # https://mbientlab.com/documents/metawear/cpp/latest/accelerometer_8h.html
    # https://mbientlab.com/documents/metawear/cpp/latest/gyro__bosch_8h.html
    'acc_odr': 100.0,  # Hz
    'acc_range': 16.0,  # G
    'gyro_odr': GyroBoschOdr._100Hz,
    'gyro_range': GyroBoschRange._1000dps,

    # Connection parameters see:
    # https://mbientlab.com/documents/metawear/cpp/latest/settings_8h.html#a1cf3cae052fe7981c26124340a41d66d
    'min_conn_interval': 7.5,
    'max_conn_interval': 7.5,
    'latency': 0,
    'timeout': 6000,

    'connection_retries': 10,  # How often the program will try to connect to the sensors before giving up.

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

networks = ["old100", "mbientlab100", "mocap_half", "mbientlab200", "mbientlab200_nonIMU", "motionminers_flw100"]
network_path = {"old100": f'..{os.sep}network.pt',
                "mbientlab100": f'..{os.sep}mbientlab100{os.sep}network.pt',
                "mocap_half": f'..{os.sep}mocap_half{os.sep}network.pt',
                "motionminers_flw100": f'..{os.sep}motionminers_flw100{os.sep}network.pt',
                "mbientlab200_nonIMU": f'..{os.sep}mbientlab200_nonIMU{os.sep}network.pt',
                "mbientlab200": f'..{os.sep}mbientlab200{os.sep}network.pt',
                }


network_window_size = {"old100": 100,
                       "mbientlab100": 100,
                       "mocap_half": 100,
                       "motionminers_flw100": 100,
                       "mbientlab200_nonIMU": 200,
                       "mbientlab200": 200,}

# Only networks with this window_size will be shown
window_size = 200

# Raw data from each sensor.
t_data = np.zeros((window_size, 6))
la_data = np.zeros((window_size, 6))
ra_data = np.zeros((window_size, 6))
ll_data = np.zeros((window_size, 6))
rl_data = np.zeros((window_size, 6))

# L2 Norm of each sensors raw data.
t_energy = np.zeros((window_size,))
la_energy = np.zeros((window_size,))
ra_energy = np.zeros((window_size,))
ll_energy = np.zeros((window_size,))
rl_energy = np.zeros((window_size,))
