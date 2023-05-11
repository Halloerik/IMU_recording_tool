"""
Created on 18.2.2020

@author: Erik Altermann, Fernando Moya Rueda, Arthur Matei
@email: erik.altermann@tu-dortmund.de, 	fernando.moya@tu-dortmund.de, arthur.matei@tu-dortmund.de
"""

import traceback
from builtins import map
from _functools import reduce
from time import sleep
import datetime
import os

from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import QTimer, QThread, pyqtSlot, pyqtSignal
import pyqtgraph as pg

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

from mbientlab.metawear.cbindings import *
from mbientlab.metawear import MetaWear, libmetawear
import mbientlab.warble

import network
from settings import settings, state, \
    t_energy, la_energy, ra_energy, ll_energy, rl_energy, \
    t_data, la_data, ra_data, ll_data, rl_data

use_fake_sensors = False  # Useful for debugging without connecting sensors
if use_fake_sensors:
    from fake_sensors import SensorThread, SensorResetThread
else:
    from sensors import SensorThread, SensorResetThread

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')


class GUI(QtWidgets.QMainWindow):
    """Main GUI Class. Sets up and provides the functions for the GUI"""

    def __init__(self):
        """Starts the GUI."""

        super(GUI, self).__init__()

        uic.loadUi('gui2.ui', self)  # Load the .ui file
        self.init_gui_elements()

        # These are the threads that control the sensors once they are connected
        self.t_sensor_thread = None
        self.la_sensor_thread = None
        self.ra_sensor_thread = None
        self.ll_sensor_thread = None
        self.rl_sensor_thread = None

        # This Timer is for checking the Batterystatus of connected Sensors
        self.battery_timer = QTimer()
        self.battery_timer.timeout.connect(self.update_battery)
        self.battery_timer.start(1000)

        # This Timer will automatically stop recordings after the amount of ms defined in settings['recording_interval']
        self.recording_timer = QTimer()
        self.recording_timer.timeout.connect(self.stop_recording)
        self.recording_timer.setSingleShot(True)
        self.show()

        # This Timer will show the recorded data of each sensor in real time as they are recording
        self.graph_timer = RecordingGraphThread(t_plot=self.t_plot,
                                                la_plot=self.la_plot, ra_plot=self.ra_plot,
                                                ll_plot=self.ll_plot, rl_plot=self.rl_plot)

        # This timer will use a neural network to analyse the recorded data live.
        # It is initialized in the self.load_network() method
        self.analysis_timer = None

    def init_gui_elements(self):
        """Connects GUI-Elements to their functions"""

        # -----------------------------------------------------
        # ----------Connection---------------------------------
        # -----------------------------------------------------

        # ---------Connection Buttons----------
        self.t_connection_button = self.findChild(QtWidgets.QPushButton, 't_connection_button')
        self.t_connection_button.clicked.connect(lambda _: self.connect_sensor('t'))
        self.ra_connection_button = self.findChild(QtWidgets.QPushButton, 'ra_connection_button')
        self.ra_connection_button.clicked.connect(lambda _: self.connect_sensor('ra'))
        self.la_connection_button = self.findChild(QtWidgets.QPushButton, 'la_connection_button')
        self.la_connection_button.clicked.connect(lambda _: self.connect_sensor('la'))
        self.rl_connection_button = self.findChild(QtWidgets.QPushButton, 'rl_connection_button')
        self.rl_connection_button.clicked.connect(lambda _: self.connect_sensor('rl'))
        self.ll_connection_button = self.findChild(QtWidgets.QPushButton, 'll_connection_button')
        self.ll_connection_button.clicked.connect(lambda _: self.connect_sensor('ll'))

        # ---------Charge Labels---------------
        self.t_charge_label = self.findChild(QtWidgets.QLabel, 't_charge_label')
        self.ra_charge_label = self.findChild(QtWidgets.QLabel, 'ra_charge_label')
        self.la_charge_label = self.findChild(QtWidgets.QLabel, 'la_charge_label')
        self.rl_charge_label = self.findChild(QtWidgets.QLabel, 'rl_charge_label')
        self.ll_charge_label = self.findChild(QtWidgets.QLabel, 'll_charge_label')

        # ---------Status Labels---------------
        self.t_status_label = self.findChild(QtWidgets.QLabel, 't_status_label')
        self.ra_status_label = self.findChild(QtWidgets.QLabel, 'ra_status_label')
        self.la_status_label = self.findChild(QtWidgets.QLabel, 'la_status_label')
        self.rl_status_label = self.findChild(QtWidgets.QLabel, 'rl_status_label')
        self.ll_status_label = self.findChild(QtWidgets.QLabel, 'll_status_label')

        # --------Other Buttons---------------
        self.record_button = self.findChild(QtWidgets.QPushButton, 'record_button')
        self.record_button.clicked.connect(lambda _: self.start_recording())
        # settings_button = self.findChild(QtWidgets.QPushButton,'settings_button')
        # settings_button.clicked.connect(lambda _: _)#TODO:
        self.reset_button = self.findChild(QtWidgets.QPushButton, 'reset_button')
        self.reset_button.clicked.connect(lambda _: self.full_reset())
        self.quit_button = self.findChild(QtWidgets.QPushButton, 'quit_button')
        self.quit_button.clicked.connect(lambda _: self.close())

        # ----------Line Edits----------------------
        self.person_lineEdit = self.findChild(QtWidgets.QLineEdit, 'person_lineEdit')
        # self.person_lineEdit

        # -----------------------------------------------------
        # ----------Recording graphs---------------------------
        # -----------------------------------------------------
        t_graph = self.findChild(pg.PlotWidget, "graphicsView_t")
        t_graph.setYRange(0, 500)
        self.t_plot = t_graph.plot(t_energy)

        la_graph = self.findChild(pg.PlotWidget, "graphicsView_la")
        la_graph.setYRange(0, 500)
        self.la_plot = la_graph.plot(la_energy)

        ra_graph = self.findChild(pg.PlotWidget, "graphicsView_ra")
        ra_graph.setYRange(0, 500)
        self.ra_plot = ra_graph.plot(ra_energy)

        ll_graph = self.findChild(pg.PlotWidget, "graphicsView_ll")
        ll_graph.setYRange(0, 500)
        self.ll_plot = ll_graph.plot(ll_energy)

        rl_graph = self.findChild(pg.PlotWidget, "graphicsView_rl")
        rl_graph.setYRange(0, 500)
        self.rl_plot = rl_graph.plot(rl_energy)

        # -----------------------------------------------------
        # ----------Analysis graphs----------------------------
        # -----------------------------------------------------
        self.c_graph = self.findChild(pg.PlotWidget, "graphicsView_c")
        self.a_graph = self.findChild(pg.PlotWidget, "graphicsView_a")

        self.nn_button = self.findChild(QtWidgets.QPushButton, "pushButton_load_nn")
        self.nn_button.clicked.connect(lambda _: self.load_network())

        self.nn_label = self.findChild(QtWidgets.QLabel, "label_nn")

    def closeEvent(self, event):
        """Asks user if window wants to close the Programm.
        Makes sure to check that all Sensors are disconnected properly before closing the Programm"""

        reply = QMessageBox.question(self, 'Window Close', 'Are you sure you want to close the window?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            state['end'] = True
            if self.t_sensor_thread is not None:
                self.t_sensor_thread.wait()
            if self.la_sensor_thread is not None:
                self.la_sensor_thread.wait()
            if self.ra_sensor_thread is not None:
                self.ra_sensor_thread.wait()
            if self.ll_sensor_thread is not None:
                self.ll_sensor_thread.wait()
            if self.rl_sensor_thread is not None:
                self.rl_sensor_thread.wait()
            event.accept()
        else:
            event.ignore()

    def connect_sensor(self, name):
        """Connects to a sensor.

        Parameters:
        name : str
            Name of the sensor to connect to.
            Possible values are 't', 'la', 'ra', 'll', 'rl'
        """

        if name == 't':
            self.t_sensor_thread = SensorThread(name)
            self.t_sensor_thread.connected.connect(self.connected_sensor)
            self.t_sensor_thread.disconnected.connect(self.disconnected_sensor)
            self.t_sensor_thread.connection_attempt.connect(lambda i: self.t_status_label.setText(
                f"Connecting {i + 1}/{settings['connection_retries']}"))
            self.t_sensor_thread.connection_failed.connect(lambda: self.t_status_label.setText("Connection failed"))
            self.t_sensor_thread.start()
            button = self.t_connection_button
        elif name == 'la':
            self.la_sensor_thread = SensorThread(name)
            self.la_sensor_thread.connected.connect(self.connected_sensor)
            self.la_sensor_thread.disconnected.connect(self.disconnected_sensor)
            self.la_sensor_thread.connection_attempt.connect(lambda i: self.la_status_label.setText(
                f"Connecting {i + 1}/{settings['connection_retries']}"))
            self.la_sensor_thread.connection_failed.connect(lambda: self.la_status_label.setText("Connection failed"))
            self.la_sensor_thread.start()
            button = self.la_connection_button
        elif name == 'ra':
            self.ra_sensor_thread = SensorThread(name)
            self.ra_sensor_thread.connected.connect(self.connected_sensor)
            self.ra_sensor_thread.disconnected.connect(self.disconnected_sensor)
            self.ra_sensor_thread.connection_attempt.connect(lambda i: self.ra_status_label.setText(
                f"Connecting {i + 1}/{settings['connection_retries']}"))
            self.ra_sensor_thread.connection_failed.connect(lambda: self.ra_status_label.setText("Connection failed"))
            self.ra_sensor_thread.start()
            button = self.ra_connection_button
        elif name == 'll':
            self.ll_sensor_thread = SensorThread(name)
            self.ll_sensor_thread.connected.connect(self.connected_sensor)
            self.ll_sensor_thread.disconnected.connect(self.disconnected_sensor)
            self.ll_sensor_thread.connection_attempt.connect(lambda i: self.ll_status_label.setText(
                f"Connecting {i + 1}/{settings['connection_retries']}"))
            self.ll_sensor_thread.connection_failed.connect(lambda: self.ll_status_label.setText("Connection failed"))
            self.ll_sensor_thread.start()
            button = self.ll_connection_button
        elif name == 'rl':
            self.rl_sensor_thread = SensorThread(name)
            self.rl_sensor_thread.connected.connect(self.connected_sensor)
            self.rl_sensor_thread.disconnected.connect(self.disconnected_sensor)
            self.rl_sensor_thread.connection_attempt.connect(lambda i: self.rl_status_label.setText(
                f"Connecting {i + 1}/{settings['connection_retries']}"))
            self.rl_sensor_thread.connection_failed.connect(lambda: self.rl_status_label.setText("Connection failed"))
            self.rl_sensor_thread.start()
            button = self.rl_connection_button
        else:
            raise ValueError

        button.setEnabled(False)

        # Disables the button until thread is finished
        # for button in [self.t_connection_button,
        #               self.la_connection_button,self.ra_connection_button,
        #               self.ll_connection_button,self.rl_connection_button]:
        #    button.setEnabled(False)

    @pyqtSlot(str)
    def connected_sensor(self, name):
        """A slot which is called once a sensor has successfully connected and finished its setup.

        Parameters:
        name : str
            Name of the sensor which was connected.
            Possible values are 't', 'la', 'ra', 'll', 'rl'
        """

        if name == 't':
            button = self.t_connection_button
            status_label = self.t_status_label
        elif name == 'la':
            button = self.la_connection_button
            status_label = self.la_status_label
        elif name == 'ra':
            button = self.ra_connection_button
            status_label = self.ra_status_label
        elif name == 'll':
            button = self.ll_connection_button
            status_label = self.ll_status_label
        elif name == 'rl':
            button = self.rl_connection_button
            status_label = self.rl_status_label
        else:
            raise ValueError

        # Changes the Connect button to be a Disconnect button.
        button.setText("Disconnect")
        button.clicked.disconnect()
        button.clicked.connect(lambda _: self.disconnect_sensor(name))
        button.setEnabled(True)

        status_label.setText("Connected")

        # for button in [self.t_connection_button,
        #               self.la_connection_button,self.ra_connection_button,
        #               self.ll_connection_button,self.rl_connection_button]:
        #    button.setEnabled(True)

    def disconnect_sensor(self, name):
        """Sends signal to Disconnect a sensor

        Parameters:
        name : str
            Name of the sensor to connect to.
            Possible values are 't','la','ra','ll','rl'
        """

        if name == 't':
            self.t_sensor_thread.disconnect_sensor()
            button = self.t_connection_button
        elif name == 'la':
            self.la_sensor_thread.disconnect_sensor()
            button = self.la_connection_button
        elif name == 'ra':
            self.ra_sensor_thread.disconnect_sensor()
            button = self.ra_connection_button
        elif name == 'll':
            self.ll_sensor_thread.disconnect_sensor()
            button = self.ll_connection_button
        elif name == 'rl':
            self.rl_sensor_thread.disconnect_sensor()
            button = self.rl_connection_button
        else:
            raise ValueError

        # Disables the button until thread is finished
        button.setEnabled(False)

    @pyqtSlot(str)
    def disconnected_sensor(self, name):
        """A slot which is called once a sensor finished the disconnection process.

        Parameters:
        name : str
            Name of the disconnected sensor.
            Possible values are 't','la','ra','ll','rl'
        """

        if name == 't':
            self.t_sensor_thread = None
            button = self.t_connection_button
            charge_label = self.t_charge_label
            status_label = self.t_status_label
        elif name == 'la':
            self.la_sensor_thread = None
            button = self.la_connection_button
            charge_label = self.la_charge_label
            status_label = self.la_status_label
        elif name == 'ra':
            self.ra_sensor_thread = None
            button = self.ra_connection_button
            charge_label = self.ra_charge_label
            status_label = self.ra_status_label
        elif name == 'll':
            self.ll_sensor_thread = None
            button = self.ll_connection_button
            charge_label = self.ll_charge_label
            status_label = self.ll_status_label
        elif name == 'rl':
            self.rl_sensor_thread = None
            button = self.rl_connection_button
            charge_label = self.rl_charge_label
            status_label = self.rl_status_label
        else:
            raise ValueError

        # Changes the disconnect Button back to a connect Button
        button.setText("Connect")
        button.clicked.disconnect()
        button.clicked.connect(lambda _: self.connect_sensor(name))
        button.setEnabled(True)

        charge_label.setText("---%")
        status_label.setText("Disconnected")

    def update_battery(self):
        """Checks the Batterycharge percentage of the Sensors and updates the GUI"""

        if self.t_sensor_thread is not None:
            charge = self.t_sensor_thread.check_battery()
            self.t_charge_label.setText('{}%'.format(charge))

        if self.la_sensor_thread is not None:
            charge = self.la_sensor_thread.check_battery()
            self.la_charge_label.setText('{}%'.format(charge))

        if self.ra_sensor_thread is not None:
            charge = self.ra_sensor_thread.check_battery()
            self.ra_charge_label.setText('{}%'.format(charge))

        if self.ll_sensor_thread is not None:
            charge = self.ll_sensor_thread.check_battery()
            self.ll_charge_label.setText('{}%'.format(charge))

        if self.rl_sensor_thread is not None:
            charge = self.rl_sensor_thread.check_battery()
            self.rl_charge_label.setText('{}%'.format(charge))

    def load_network(self):
        """Loads the neural network used for live analysis and initializes the analysis thread"""
        net = network.load_network("../network.pt")
        self.nn_label.setText("Loaded")

        attribute_rep = np.loadtxt(f"..{os.sep}atts_per_class_dataset.txt", delimiter=",")

        self.nn_button.clicked.disconnect()
        self.nn_button.clicked.connect(lambda _: self.unload_network())
        self.nn_button.setText("Unload")

        self.analysis_timer = AnalysisThread(net, attribute_rep, self.a_graph, self.c_graph)

    def unload_network(self):
        """Unloads the neural network and stops the analysis thread"""
        self.analysis_timer.stop_timer()
        self.analysis_timer = None

        self.nn_label.setText("None")

        self.nn_button.clicked.disconnect()
        self.nn_button.clicked.connect(lambda _: self.load_network())
        self.nn_button.setText("Load")

    def start_recording(self):
        """Starts the Recording"""

        # List of all sensor_threads
        threads = [self.t_sensor_thread, self.la_sensor_thread, self.ra_sensor_thread, self.ll_sensor_thread,
                   self.rl_sensor_thread]

        # Checks if all Sensors are connected before a recording starts
        if all(map(lambda x: x is not None, threads)):
            # if any(map(lambda x: x is not None, threads)):

            # Determine path for storing sensor data
            person = self.person_lineEdit.text()
            time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            path = "{}{}{}{}{}".format(settings['folder'], os.sep, person, os.sep, time)
            if not os.path.exists(path):
                os.makedirs(path)
            state['final_folder'] = path
            sleep(1)

            # Disable all Disconnect Buttons so that sensors can't change during Recording
            self.t_connection_button.setEnabled(False)
            self.la_connection_button.setEnabled(False)
            self.ra_connection_button.setEnabled(False)
            self.ll_connection_button.setEnabled(False)
            self.rl_connection_button.setEnabled(False)
            self.reset_button.setEnabled(False)

            self.nn_button.setEnabled(False)

            # Update status labels
            self.t_status_label.setText("Recording")
            self.la_status_label.setText("Recording")
            self.ra_status_label.setText("Recording")
            self.ll_status_label.setText("Recording")
            self.rl_status_label.setText("Recording")

            # Turn start-recording-button into stop-recording-button
            self.record_button.setText("Stop recording")
            self.record_button.clicked.disconnect()
            self.record_button.clicked.connect(lambda _: self.stop_recording())

            # Start recording and the recording timer.
            state['recording'] = True
            print('recording!')
            self.recording_timer.start(settings['recording_interval'])
            self.graph_timer.start_timer()
            if self.analysis_timer:
                self.analysis_timer.start_timer()
            sleep(1)
        else:
            print('Cant start recording. At least one sensor is not connected.')

    def stop_recording(self):
        """Stops the Recording"""

        # Stops the timer so that this Function doesn't get excecuted twice if the user stops the recording manually
        self.recording_timer.stop()
        self.graph_timer.stop_timer()
        if self.analysis_timer:
            self.analysis_timer.stop_timer()

        # Enables all connection buttons again
        self.t_connection_button.setEnabled(True)
        self.la_connection_button.setEnabled(True)
        self.ra_connection_button.setEnabled(True)
        self.ll_connection_button.setEnabled(True)
        self.rl_connection_button.setEnabled(True)
        self.reset_button.setEnabled(True)

        self.nn_button.setEnabled(True)

        # Update status labels #TODO make this show the number of recorded samples?
        self.t_status_label.setText("Ready")
        self.la_status_label.setText("Ready")
        self.ra_status_label.setText("Ready")
        self.ll_status_label.setText("Ready")
        self.rl_status_label.setText("Ready")

        # Turn stop-recording-button back into start-recording-button
        self.record_button.setText("Start recording")
        self.record_button.clicked.disconnect()
        self.record_button.clicked.connect(lambda _: self.start_recording())

        # Stops the recording
        state['recording'] = False
        print('recording stopped')
        sleep(1)

    def full_reset(self):
        """Disconnects and resets sensors in case of an error"""

        threads = [self.t_sensor_thread, self.la_sensor_thread, self.ra_sensor_thread,
                   self.ll_sensor_thread, self.rl_sensor_thread]
        buttons = [self.t_connection_button, self.la_connection_button, self.ra_connection_button,
                   self.ll_connection_button, self.rl_connection_button, self.reset_button, self.record_button]

        for button in buttons:
            button.setEnabled(False)

        # Disconnects all sensors
        print("disconnecting from sensors")
        state['recording'] = False
        state['end'] = True
        for thread in threads:
            if thread is not None:
                thread.wait()
        state['end'] = False

        # Buttons would get enabled again by the disconnected_sensor method after the threads quit
        for button in buttons:
            button.setEnabled(False)  # TODO race condition with threads ending causes the buttons to be enabled anyway

        # Resets all sensors
        print("resetting sensors")

        self.t_sensor_thread = SensorResetThread('t')
        self.la_sensor_thread = SensorResetThread('la')
        self.ra_sensor_thread = SensorResetThread('ra')
        self.ll_sensor_thread = SensorResetThread('ll')
        self.rl_sensor_thread = SensorResetThread('rl')

        threads = [self.t_sensor_thread, self.la_sensor_thread, self.ra_sensor_thread,
                   self.ll_sensor_thread, self.rl_sensor_thread]

        status_labels = [self.t_status_label, self.la_status_label, self.ra_status_label,
                         self.ll_status_label, self.rl_status_label]

        for thread, label in zip(threads, status_labels):
            thread.reset_attempt.connect(lambda i, l=label: l.setText(
                f"\rResetting {i + 1}/{settings['connection_retries']}"))
            thread.reset_success.connect(lambda _, l=label: l.setText("Reset successful"))
            thread.reset_success.connect(self.reset_finished)
            thread.reset_failed.connect(lambda _, l=label: l.setText("Reset failed"))
            thread.reset_failed.connect(self.reset_finished)
            thread.start()

    @pyqtSlot(str)
    def reset_finished(self, name):
        """A slot which is called once a sensor finished the reset process.

        Parameters:
        name : str
            Name of the disconnected sensor.
            Possible values are 't','la','ra','ll','rl'
        """

        buttons = [self.t_connection_button, self.la_connection_button, self.ra_connection_button,
                   self.ll_connection_button, self.rl_connection_button, self.reset_button, self.record_button]

        # Dereference the sensor_thread that called this method
        if name == 't':
            self.t_sensor_thread = None
        elif name == 'la':
            self.la_sensor_thread = None
        elif name == 'ra':
            self.ra_sensor_thread = None
        elif name == 'll':
            self.ll_sensor_thread = None
        elif name == 'rl':
            self.rl_sensor_thread = None
        else:
            raise ValueError

        threads = [self.t_sensor_thread, self.la_sensor_thread, self.ra_sensor_thread,
                   self.ll_sensor_thread, self.rl_sensor_thread]

        # If all threads are finished, enable the buttons
        if all(map(lambda x: x is None, threads)):
            for button in buttons:
                button.setEnabled(True)


class RecordingGraphThread(QThread):
    """Thread for showing recorded data live"""

    # on_timeout = pyqtSignal()

    def __init__(self, t_plot: pg.PlotItem, la_plot: pg.PlotItem, ra_plot: pg.PlotItem,
                 ll_plot: pg.PlotItem, rl_plot: pg.PlotItem):
        """"""
        super(RecordingGraphThread, self).__init__()

        self.t_plot = t_plot
        self.la_plot = la_plot
        self.ra_plot = ra_plot
        self.ll_plot = ll_plot
        self.rl_plot = rl_plot

        self.timer = 0
        self.interval = settings["graph_interval"]

    def update_graphs(self):
        """"""
        # print("recording graph update graphs")
        t_energy[:-1] = t_energy[1:]
        t_energy[-1] = np.linalg.norm(t_data[-1])
        la_energy[:-1] = la_energy[1:]
        la_energy[-1] = np.linalg.norm(la_data[-1])
        ra_energy[:-1] = ra_energy[1:]
        ra_energy[-1] = np.linalg.norm(ra_data[-1])
        ll_energy[:-1] = ll_energy[1:]
        ll_energy[-1] = np.linalg.norm(ll_data[-1])
        rl_energy[:-1] = rl_energy[1:]
        rl_energy[-1] = np.linalg.norm(rl_data[-1])

        self.t_plot.setData(t_energy)
        self.la_plot.setData(la_energy)
        self.ra_plot.setData(ra_energy)
        self.ll_plot.setData(ll_energy)
        self.rl_plot.setData(rl_energy)

    def start_timer(self):
        # print("recording graph starting timer")
        self.timer = self.startTimer(self.interval)
        # print("timer started, id:",self.timer)

    def stop_timer(self):
        self.killTimer(self.timer)
        self.timer = 0

    def set_interval(self, interval):
        self.interval = interval
        if self.timer != 0:  # If already running: stop and start with new value.
            self.stop_timer()
            self.start_timer()

    def timerEvent(self, _):  # Timer gets discarded. There is only 1 active timer anyway.
        # self.on_timeout.emit()
        self.update_graphs()
        # print("recording graph timer event")


class AnalysisThread(QThread):
    """Thread for analysing recorded data live with a neural network"""

    # on_timeout = pyqtSignal()

    def __init__(self, network: network.Network, attribute_representation: np.ndarray,
                 a_graph: pg.PlotItem, c_graph: pg.PlotItem):

        super().__init__()
        self.timer = 0
        self.interval = settings["analysis_interval"]

        self.network = network
        self.attribute_representation = attribute_representation

        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.data = torch.randn((1, 1, 30, 200)).to(device=self.device)  # data.shape -> [batch, 1, Channels, Time]

        self.a_graph = a_graph
        self.c_graph = c_graph

        self.class_bars = []
        self.attribute_bars = []

        # Setup Attribute graph
        self.a_graph.clear()
        self.a_graph.getAxis('left').setLabel(text='Attributes', units='')
        self.a_graph.setYRange(0, 1, padding=0.1)
        with open("../attrib.txt", "r") as f:
            attributes = f.read().split(",")
            for i, attribute in enumerate(attributes):
                bar = pg.BarGraphItem(x0=[i], x1=i + 1, y0=0, y1=0, name=attribute)
                self.attribute_bars.append(bar)
                self.a_graph.addItem(bar)

                label = pg.TextItem(text=attribute, color='b', anchor=(0, 0), border=None, fill=None, angle=-90,
                                    rotateAxis=None)
                label.setPos(i + 1, 1)

                self.a_graph.addItem(label)
            self.a_graph.setXRange(0, len(attributes), padding=0.02)

        # Setup Class graph
        self.c_graph.clear()
        self.c_graph.getAxis('left').setLabel(text='Classes', units='')
        self.c_graph.getAxis('bottom').setLabel(text='Time', units='s')
        windows_per_recording = settings["recording_interval"] // settings['analysis_interval']
        ticks = [(0, "now")]
        ticks.extend([(-i, f"{-i}") for i in range(20, windows_per_recording + 1, 20)])
        self.c_graph.getAxis('bottom').setTicks([ticks])
        self.classes = np.zeros(windows_per_recording)
        self.c_graph.setXRange(-windows_per_recording - 5, 5, padding=0.02)
        with open("../class.txt", "r") as f:
            self.class_names = f.read().split(",")
            self.c_graph.setYRange(0, len(self.class_names) + 1, padding=0)

            self.current_class_label = pg.TextItem(text="Current Class:", color='b', anchor=(1, 4), border=None,
                                                   fill=None, angle=0, rotateAxis=None)
            self.c_graph.addItem(self.current_class_label)

            for i in range(windows_per_recording):
                bar = pg.BarGraphItem(x0=[i - windows_per_recording], x1=i + 1 - windows_per_recording, y0=0, y1=0,
                                      name=i)
                self.class_bars.append(bar)
                self.c_graph.addItem(bar)

    def start_timer(self):
        self.timer = self.startTimer(self.interval)
        # print("timer started, id:",self.timer)

    def stop_timer(self):
        self.killTimer(self.timer)
        self.timer = 0
        # print("timer stopped")

    def set_interval(self, interval):
        self.interval = interval
        if self.timer != 0:  # If already running: stop and start with new value.
            self.stop_timer()
            self.start_timer()

    def timerEvent(self, _):  # Timer gets discarded. There is only 1 active timer anyway.
        # self.on_timeout.emit()
        self.analyse_data()

    def analyse_data(self):
        # get data from settings.py
        self.data[0, 0, 0:6, :] = torch.from_numpy(la_data.T)
        self.data[0, 0, 6:12, :] = torch.from_numpy(ll_data.T)
        self.data[0, 0, 12:18, :] = torch.from_numpy(t_data.T)
        self.data[0, 0, 18:24, :] = torch.from_numpy(ra_data.T)
        self.data[0, 0, 24:30, :] = torch.from_numpy(rl_data.T)

        self.data = network.norm_mbientlab(self.data).to(device=self.device)

        # pass data through network
        result = self.network(self.data).detach().cpu().numpy()
        # calculate attributes and classes
        cls, attr = self.predict_class_and_attributes(result)
        # print(result.shape, cls.shape, attr.shape)
        # print(result, cls, "\n", attr)

        # update attribute graph
        for i in range(len(self.attribute_bars)):
            self.attribute_bars[i].setOpts(y1=attr[i])

        # update class graph
        self.current_class_label.setText(f"Current Class: {self.class_names[cls]}")
        self.classes[:-1] = self.classes[1:]
        self.classes[-1] = cls
        for i in range(len(self.class_bars)):
            self.class_bars[i].setOpts(y1=self.classes[i])

    def predict_class_and_attributes(self, result):
        attributes = np.array(self.attribute_representation[:, 1:])
        classes = np.array(self.attribute_representation[:, 0])

        distances = 1 - cosine_similarity(result, attributes)

        sorted_distances = np.argsort(distances, 1)

        sorted_attributes = attributes[sorted_distances[0]]
        sorted_classes = classes[sorted_distances[0]]

        return int(sorted_classes[0].item()), sorted_attributes[0]
