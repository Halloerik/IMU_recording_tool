"""
Created on 20.04.2023

@author: Erik Altermann, Fernando Moya Rueda, Arthur Matei
@email: erik.altermann@tu-dortmund.de, 	fernando.moya@tu-dortmund.de, arthur.matei@tu-dortmund.de
"""

from __future__ import print_function

import traceback

from time import sleep
from threading import Event
import datetime
import csv
from PyQt5 import QtCore
from settings import settings, state, \
    t_data, la_data, ra_data, ll_data, rl_data, \
    t_energy, la_energy, ra_energy, ll_energy, rl_energy
import os
import numpy as np


class SensorState:
    """Handles the connection to the sensor"""

    def __init__(self, device, name):
        """Initializes the state

        Parameters
        device
            the object returned by MetaWear(address)
        name : str
            the name of this sensor
        """

        self.device = device
        self.name = name
        self.samples = 0

        self.charge = 'fake'

    def data_handler(self):
        """Saves the newest recieved"""

        if self.name == 't':
            t_data[:-1] = t_data[1:]
            t_data[-1] = self.data[self.data_index]
        elif self.name == 'la':
            la_data[:-1] = la_data[1:]
            la_data[-1] = self.data[self.data_index]
        elif self.name == 'ra':
            ra_data[:-1] = ra_data[1:]
            ra_data[-1] = self.data[self.data_index]
        elif self.name == 'll':
            ll_data[:-1] = ll_data[1:]
            ll_data[-1] = self.data[self.data_index]
        elif self.name == 'rl':
            rl_data[:-1] = rl_data[1:]
            rl_data[-1] = self.data[self.data_index]
        else:
            raise ValueError

        self.data_index = 0 if self.data_index + 1 == self.data.shape[0] else self.data_index + 1

    def setup(self):
        """sets up the datasignals and sensor parameters"""

        data = np.load("../test_data.npy")
        if self.name == 't':
            self.data = data[:, 0:6]
        elif self.name == 'la':
            self.data = data[:, 6:12]
        elif self.name == 'ra':
            self.data = data[:, 12:18]
        elif self.name == 'll':
            self.data = data[:, 18:24]
        elif self.name == 'rl':
            self.data = data[:, 24:30]
        else:
            raise ValueError

        self.data_index = 0

    def start_recording(self):
        """starts the recording"""

    def stop_recording(self):
        """stops the recording"""

    def disconnect(self, error_dc=False):
        """Disconnects the sensor"""
        sleep(2)


class SensorThread(QtCore.QThread):
    """Handles the Sensor state"""

    connected = QtCore.pyqtSignal(str)
    disconnected = QtCore.pyqtSignal(str)
    connection_attempt = QtCore.pyqtSignal(int)
    connection_failed = QtCore.pyqtSignal()

    def __init__(self, name):
        """Initializes a Connection to the Sensor
        Parameters:
        name : str
            Name of the sensor to connect to.
            Possible values are 't','la','ra','ll','rl'
        """

        super(SensorThread, self).__init__()
        self.name = name

        # Mac address of the needed sensor
        self.address = settings[name + '_address']
        self.disconnect = False
        self.recording = False

        self.state = None

    def run(self):
        """main loop of the thread."""

        # Connection to the needed sensor
        print("connecting to {}".format(self.address))
        for i in range(settings['connection_retries']):
            print(f"\r  attempt {i + 1}/{settings['connection_retries']}", end="")
            self.connection_attempt.emit(i)
            sleep(1)
            self.connected.emit(self.name)
            print(" connected")
            break

        # sets up the sensor and its signals/data handling
        self.state = SensorState(device=None, name=self.name)
        self.state.setup()

        while True:
            # Checks if sensor should be disconnected
            if state['end'] or self.disconnect:
                if state['recording'] or self.recording:
                    self.state.stop_recording()
                self.state.disconnect()
                break
            # Checks if recording should be started. If True and not already recording starts the recording.
            if state['recording'] and not self.recording:
                self.state.start_recording()
                self.recording = True
            # Checks if ongoing recording should be stopped. If True it stops the recording
            if self.recording and not state['recording']:
                self.state.stop_recording()
                self.recording = False
            sleep(0.010)
            if state['recording']:
                self.state.data_handler()

        # self.device.disconnect() this already happens in the self.state.disconnect() method
        sleep(2)
        self.disconnected.emit(self.name)
        self.quit()

    def disconnect_sensor(self):
        self.disconnect = True

    def check_battery(self):
        """Checks battery and returns the value."""
        return "fake"


class SensorResetThread(QtCore.QThread):
    """Resets the sensor"""

    reset_attempt = QtCore.pyqtSignal(int)
    reset_success = QtCore.pyqtSignal(str)
    reset_failed = QtCore.pyqtSignal(str)

    def __init__(self, name):
        """Initializes a Connection to the Sensor
        Parameters:
        name : str
            Name of the sensor to connect to.
            Possible values are 't','la','ra','ll','rl'
        """

        super(SensorResetThread, self).__init__()
        self.name = name

    def run(self):
        """main loop of the thread."""

        # Connection to the needed sensor
        for i in range(settings['connection_retries']):
            self.reset_attempt.emit(i)
            sleep(5)
            self.reset_success.emit(self.name)
            break

        self.quit()

    def check_battery(self):
        return "---"
