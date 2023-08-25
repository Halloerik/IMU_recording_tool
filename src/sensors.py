"""
Created on 18.2.2020

@author: Erik Altermann, Fernando Moya Rueda, Arthur Matei
@email: erik.altermann@tu-dortmund.de, 	fernando.moya@tu-dortmund.de, arthur.matei@tu-dortmund.de
"""

from __future__ import print_function

import traceback

from mbientlab.metawear import MetaWear, libmetawear, parse_value
from mbientlab.metawear.cbindings import *
import mbientlab.warble

from time import sleep
from threading import Event
import datetime
import csv
from PyQt5 import QtCore
from settings import settings, state, t_data, la_data, ra_data, ll_data, rl_data
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
        self.callback = FnVoid_VoidP_DataP(self.data_handler)
        self.callback_battery = FnVoid_VoidP_DataP(self.battery_handler)
        self.battery = libmetawear.mbl_mw_settings_get_battery_state_data_signal(self.device.board)
        self.charge = '---'
        self.processor = None

    def data_handler(self, ctx, data):
        """Saves the newest recieved"""

        self.samples += 1
        # print(data,data.contents,"\n", data["epoch"],data["extra"],data["value"])


        values = parse_value(data, n_elem=2)
        self.writer.writerow((self.name, datetime.datetime.now(), str(values[0].x), str(values[0].y), str(values[0].z),
                              str(values[1].x), str(values[1].y), str(values[1].z)))

        arr = np.array((values[0].x, values[0].y, values[0].z, values[1].x, values[1].y, values[1].z))
        if self.name == 't':
            t_data[:-1] = t_data[1:]
            t_data[-1] = arr
        elif self.name == 'la':
            la_data[:-1] = la_data[1:]
            la_data[-1] = arr
        elif self.name == 'ra':
            ra_data[:-1] = ra_data[1:]
            ra_data[-1] = arr
        elif self.name == 'll':
            ll_data[:-1] = ll_data[1:]
            ll_data[-1] = arr
        elif self.name == 'rl':
            rl_data[:-1] = rl_data[1:]
            rl_data[-1] = arr
        else:
            raise ValueError

    def create_file(self):
        """Creates a new csv file to save all sensor data"""

        # Creating file
        filepath = f"{state['final_folder']}{os.sep}{self.name}.csv"
        file = open(filepath, 'wt', newline='')
        self.writer = csv.writer(file, delimiter=',')
        self.writer.writerow(("IMU", "Time", "AccelerometerX", "AccelerometerY", "AccelerometerZ", "GyroscopeX",
                              "GyroscopeY", "GyroscopeZ"))
        self.file = file

    def close_file(self):
        self.file.close()

    def battery_handler(self, ctx, data):
        """Saves the Battery signal recieved after using check_battery"""

        # print("%s -> %s" % (self.device.address, parse_value(data)))
        # print(parse_value(data).charge)
        # self.samples+= 1
        self.charge = parse_value(data).charge

    def check_battery(self):
        """Reads the battery. Value gets sent to the battery_handler"""

        # self.battery = libmetawear.mbl_mw_settings_get_battery_state_data_signal(self.device.board)
        libmetawear.mbl_mw_datasignal_read(self.battery)

    def set_light_pattern(self, pattern):
        """sets a certain pattern for the LED

        Parameters:
        pattern : str
            'connected' is a pulsing orange light
            'setup' is a solid green light
            'recording' is a solid blue light
            'error' is a solid red light
            Any other argument just disables the light completely
        """

        libmetawear.mbl_mw_led_stop_and_clear(self.device.board)
        sleep(1.0)

        if pattern == 'connected':
            pattern = LedPattern(repeat_count=Const.LED_REPEAT_INDEFINITELY)
            libmetawear.mbl_mw_led_load_preset_pattern(byref(pattern), LedPreset.PULSE)
            libmetawear.mbl_mw_led_write_pattern(self.device.board, byref(pattern), LedColor.GREEN)
            libmetawear.mbl_mw_led_write_pattern(self.device.board, byref(pattern), LedColor.RED)
        elif pattern == 'setup':
            pattern = LedPattern(repeat_count=Const.LED_REPEAT_INDEFINITELY)
            libmetawear.mbl_mw_led_load_preset_pattern(byref(pattern), LedPreset.SOLID)
            libmetawear.mbl_mw_led_write_pattern(self.device.board, byref(pattern), LedColor.GREEN)
        elif pattern == 'recording':
            pattern = LedPattern(repeat_count=Const.LED_REPEAT_INDEFINITELY)
            libmetawear.mbl_mw_led_load_preset_pattern(byref(pattern), LedPreset.SOLID)
            libmetawear.mbl_mw_led_write_pattern(self.device.board, byref(pattern), LedColor.BLUE)
        elif pattern == 'error':
            pattern = LedPattern(repeat_count=Const.LED_REPEAT_INDEFINITELY)
            libmetawear.mbl_mw_led_load_preset_pattern(byref(pattern), LedPreset.SOLID)
            libmetawear.mbl_mw_led_write_pattern(self.device.board, byref(pattern), LedColor.RED)
        libmetawear.mbl_mw_led_play(self.device.board)

    def setup(self):
        """sets up the datasignals and sensor parameters"""
        print("  configuring device")
        libmetawear.mbl_mw_settings_set_connection_parameters(self.device.board,
                                                              settings['min_conn_interval'],
                                                              settings['max_conn_interval'], settings['latency'],
                                                              settings['timeout'])
        sleep(1.5)

        # Accelerometer setup
        libmetawear.mbl_mw_acc_set_range(self.device.board, settings['acc_range'])
        libmetawear.mbl_mw_acc_set_odr(self.device.board, settings['acc_odr'])
        libmetawear.mbl_mw_acc_write_acceleration_config(self.device.board)

        # Gyroscope setup
        libmetawear.mbl_mw_gyro_bmi160_set_range(self.device.board, settings['gyro_range'])
        libmetawear.mbl_mw_gyro_bmi160_set_odr(self.device.board, settings['gyro_odr'])
        libmetawear.mbl_mw_gyro_bmi160_write_config(self.device.board)

        # Subscription is handled below with the data fusion
        # acc = libmetawear.mbl_mw_acc_get_acceleration_data_signal(self.device.board)
        # libmetawear.mbl_mw_datasignal_subscribe(acc, None, self.callback)
        # gyro = libmetawear.mbl_mw_gyro_bmi160_get_rotation_data_signal(self.device.board)
        # libmetawear.mbl_mw_datasignal_subscribe(gyro, None, self.callback)

        # Battery setup
        self.battery = libmetawear.mbl_mw_settings_get_battery_state_data_signal(self.device.board)
        libmetawear.mbl_mw_datasignal_subscribe(self.battery, None, self.callback_battery)

        # Setup datafuser
        e = Event()

        def processor_created(context, pointer):
            self.processor = pointer
            e.set()

        fn_wrapper = FnVoid_VoidP_VoidP(processor_created)
        acc = libmetawear.mbl_mw_acc_get_acceleration_data_signal(self.device.board)
        gyro = libmetawear.mbl_mw_gyro_bmi160_get_rotation_data_signal(self.device.board)

        signals = (c_void_p * 1)()
        signals[0] = gyro
        libmetawear.mbl_mw_dataprocessor_fuser_create(acc, signals, 1, None, fn_wrapper)
        e.wait()
        libmetawear.mbl_mw_datasignal_subscribe(self.processor, None, self.callback)

        """libmetawear.mbl_mw_sensor_fusion_set_mode(self.device.board, SensorFusionMode.IMU_PLUS)
        libmetawear.mbl_mw_sensor_fusion_set_acc_range(self.device.board, SensorFusionAccRange._8G)
        libmetawear.mbl_mw_sensor_fusion_set_gyro_range(self.device.board, SensorFusionGyroRange._250DPS)
        libmetawear.mbl_mw_sensor_fusion_write_config(self.device.board)

        signal = libmetawear.mbl_mw_sensor_fusion_get_data_signal(self.device.board, SensorFusionData.QUATERNION)
        libmetawear.mbl_mw_datasignal_subscribe(signal, None, self.callback)

        libmetawear.mbl_mw_sensor_fusion_enable_data(self.device.board, SensorFusionData.QUATERNION)"""

        self.set_light_pattern('setup')
        print('  done')

    def start_recording(self):
        """starts the recording"""

        # Creates a new file to save the recorded data
        self.create_file()
        libmetawear.mbl_mw_acc_enable_acceleration_sampling(self.device.board)
        libmetawear.mbl_mw_gyro_bmi160_enable_rotation_sampling(self.device.board)
        libmetawear.mbl_mw_acc_start(self.device.board)
        libmetawear.mbl_mw_gyro_bmi160_start(self.device.board)

        # libmetawear.mbl_mw_sensor_fusion_start(self.device.board)

        self.set_light_pattern('recording')

    def stop_recording(self):
        """stops the recording"""
        libmetawear.mbl_mw_acc_stop(self.device.board)
        libmetawear.mbl_mw_gyro_bmi160_stop(self.device.board)
        libmetawear.mbl_mw_acc_disable_acceleration_sampling(self.device.board)
        libmetawear.mbl_mw_gyro_bmi160_disable_rotation_sampling(self.device.board)

        # libmetawear.mbl_mw_sensor_fusion_stop(self.device.board)
        self.set_light_pattern('setup')
        self.close_file()

    def disconnect(self, error_dc=False):
        """Disconnects the sensor"""

        # Unsubscribe the datasignals
        try:
            acc = libmetawear.mbl_mw_acc_get_acceleration_data_signal(self.device.board)
            libmetawear.mbl_mw_datasignal_unsubscribe(acc)
        except Exception:
            traceback.print_exc()
        try:
            gyro = libmetawear.mbl_mw_gyro_bmi160_get_rotation_data_signal(self.device.board)
            libmetawear.mbl_mw_datasignal_unsubscribe(gyro)
        except Exception:
            traceback.print_exc()
        try:
            libmetawear.mbl_mw_datasignal_unsubscribe(self.processor)
        except OSError:
            pass
        except Exception:
            traceback.print_exc()
        try:
            # battery = libmetawear.mbl_mw_settings_get_battery_state_data_signal(self.device.board)
            libmetawear.mbl_mw_datasignal_unsubscribe(self.battery)
        except Exception:
            traceback.print_exc()

        # Disable the LED
        self.set_light_pattern(None)
        sleep(2)

        # Reset everything
        libmetawear.mbl_mw_logging_stop(self.device.board)
        libmetawear.mbl_mw_logging_clear_entries(self.device.board)
        libmetawear.mbl_mw_macro_erase_all(self.device.board)
        libmetawear.mbl_mw_debug_reset_after_gc(self.device.board)
        libmetawear.mbl_mw_debug_disconnect(self.device.board)
        sleep(2)
        self.device.disconnect()
        print(f"sensor {self.name} disconnected", "\n",
              f"  address: {self.device.address}", "\n",
              f"  samples:{self.samples}")
        sleep(5)


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
            try:
                print(f"\r  attempt {i + 1}/{settings['connection_retries']}", end="")
                self.connection_attempt.emit(i)
                self.device = MetaWear(self.address)
                self.device.connect()
            except mbientlab.warble.WarbleException as e:
                if i + 1 == settings['connection_retries']:
                    traceback.print_exc()
                    self.disconnect = True
                    # self.state.disconnect(error_dc=True)
                    self.disconnected.emit(self.name)
                    self.connection_failed.emit()
                    self.quit()
                    return
                sleep(1)
            else:
                self.connected.emit(self.name)
                print(" connected")
                break

        # Sets a blinking orange light pattern for the sensor
        self.state = SensorState(self.device, self.name)
        self.state.set_light_pattern('connected')

        # sets up the sensor and its signals/data handling
        try:
            self.state.setup()
        except:
            traceback.print_exc()
            self.disconnect = True
            self.state.disconnect(error_dc=True)
            self.disconnected.emit(self.name)
            self.quit()
            return

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
            sleep(0.001)
        # self.device.disconnect() this already happens in the self.state.disconnect() method
        sleep(2)
        self.disconnected.emit(self.name)
        self.quit()

    def disconnect_sensor(self):
        self.disconnect = True

    def check_battery(self):
        """Checks battery and returns the value."""
        if not self.disconnect and self.state:
            self.state.check_battery()
            return self.state.charge
        return "---"


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
            try:
                self.reset_attempt.emit(i)
                device = MetaWear(settings[self.name + '_address'])
                device.connect()
                sleep(2)
                libmetawear.mbl_mw_logging_stop(device.board)
                libmetawear.mbl_mw_logging_clear_entries(device.board)
                libmetawear.mbl_mw_macro_erase_all(device.board)
                libmetawear.mbl_mw_debug_reset_after_gc(device.board)
                libmetawear.mbl_mw_debug_disconnect(device.board)
                sleep(2)
                device.disconnect()
                sleep(2)
                self.reset_success.emit(self.name)
                break
            except mbientlab.warble.WarbleException as e:
                if i + 1 == settings['connection_retries']:
                    self.reset_failed.emit(self.name)
                    traceback.print_exc()
                    break
        self.quit()

    def check_battery(self):
        return "---"
