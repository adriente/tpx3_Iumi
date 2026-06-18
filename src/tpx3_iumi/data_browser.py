import numpy as np

from qtpy import QtWidgets, QtCore
from pymodaq_data.data import DataRaw, Axis
from pymodaq_gui.utils.utils import mkQApp
from pymodaq_gui.utils.custom_app import CustomApp, Dock, DockArea

from custom_viewer import CustomViewer3D
from data_generator import DataGenerator

# import objgraph
import gc


class DataBrowser(CustomViewer3D) :
    callback_signal = QtCore.Signal()
    def __init__(self, area):
        super().__init__(area)
        self.controller = DataGenerator()
        self.x_axis = Axis('x axis', units='nm', data=self.controller.x, index=0)
        self.y_axis = Axis('y axis', units='nm', data=self.controller.y, index=1)
        self.e_axis = Axis('e axis', units='eV', data=self.controller.e, index=2)
        self._data = DataRaw('Raw data',
                      data=[np.zeros((self.x_axis.size,
                                      self.y_axis.size,
                                      self.e_axis.size))],
                      nav_indexes=(0, 1),
                      axes=[self.x_axis,
                            self.y_axis,
                            self.e_axis])
        self.callback = MyCallback(self.controller)
        self.callback_thread = QtCore.QThread()
        self.callback.moveToThread(self.callback_thread)
        self.callback.data_sig.connect(self.show_data)  # when the wait for acquisition returns (with data taken), emit_data will be fired

        self.callback_signal.connect(self.callback.run_acq)
        self.callback_thread.callback = self.callback
        self.callback_thread.start()

    def start(self) :
        self.controller.status = True
        self.callback_signal.emit()

    def stop(self) : 
        self.controller.status = False

    def show_data(self,data : np.ndarray) -> None :
        # dwa = DataRaw('Raw data',
        #               data=[data.copy()],
        #               nav_indexes=(0, 1),
        #               axes=[self.x_axis,
        #                     self.y_axis,
        #                     self.e_axis])
        self._show_data(data=data)
        # gc.collect()

    # def selected_1d_region_changed(self):
    #     return super().selected_1d_region_changed()

class MyCallback(QtCore.QObject) :
    data_sig = QtCore.Signal(np.ndarray)
    def __init__(self, controller : DataGenerator) :
        super(MyCallback, self).__init__()
        self.controller = controller

    def run_acq(self) :
        i = 0
        while self.controller.wait_for_acq() :
            new_data = self.controller.generate_data(i)
            self.data_sig.emit(new_data)
            i+=1

def main():
    app = mkQApp('DataPicker')
    area = DockArea()
    win = QtWidgets.QMainWindow()
    win.setCentralWidget(area)
    data_picker = DataBrowser(area)
    win.show()
    app.exec()

if __name__ == '__main__' :
    main()