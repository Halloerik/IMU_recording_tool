"""
Created on 18.2.2020

@author: Erik Altermann, Fernando Moya Rueda, Arthur Matei
@email: erik.altermann@tu-dortmund.de, 	fernando.moya@tu-dortmund.de, arthur.matei@tu-dortmund.de
"""

import sys
from PyQt5 import QtWidgets
from gui import GUI


# from sensors import SensorThread
def except_hook(cls, exception, traceback):
    sys.__excepthook__(cls, exception, traceback)


if __name__ == '__main__':
    sys.excepthook = except_hook
    # Create the Qt Application
    app = QtWidgets.QApplication(sys.argv)
    main_window = GUI()
    app.exec_()
    sys.exit(0)
