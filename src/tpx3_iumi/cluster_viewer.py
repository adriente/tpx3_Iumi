from typing import Optional
import numpy as np
import os
from numba import njit
from pymodaq_gui.plotting.data_viewers.viewer2D import Viewer2D
from qtpy import QtWidgets, QtCore
from pymodaq_gui.utils.custom_app import CustomApp, Dock, DockArea
from qtpy.QtCore import QObject, Slot
from pymodaq_data.data import (Axis, DataToExport, DataFromRoi, DataRaw,
                               DataDistribution, DataWithAxes)
from pymodaq_gui.utils.utils import mkQApp
from pymodaq_gui.plotting.items.roi import RoiInfo
from pathlib import Path
from data_processing import clusterize_single_chunk

parent = Path().resolve().parent

# TODO
# Remove that
# python -m pymodaq_gui.examples.material_icon_ex

@njit
def populate_image(to_fill : np.ndarray,data : np.ndarray) :
    for i in data.T :
        to_fill[i[2], i[3]] += 1

@njit
def populate_clusters(to_fill : np.ndarray,data : np.ndarray) :
    for i in data.T :
        if i[0] > 0 : 
            to_fill[i[0],i[1]] += 1

class ClusterViewer(CustomApp) :
    """
    This object deals with the display of 3D data with a 2D plotitem and a 1D plotitem. 
    """

    params = [
        {'title': 'File settings:', 'name': 'file_settings', 'type': 'group', 'children': [
            {'title': 'Currently opened file', 'name': 'file_path', 'type': 'str', 'readonly' : True,
             'value': ''}]},
        {'title': 'Time integration settings:', 'name': 'time_settings', 'type': 'group', 'children': [
            {'title': 'Integrated duration (ns)', 'name': 'duration', 'type': 'int',
             'value': 400},
        ]},
    ]

    def __init__(self, area: DockArea) :
        super().__init__(area)
        self.viewer2d: Optional[Viewer2D] = None
        self.setup_ui()
        self._memmap = None
        self._data = None
        self._file_path = None
        self._label_count = 1
        x_axis = Axis('x axis', units='px', data=np.linspace(start = 0.0, stop = 512, num = 512), index=0)
        y_axis = Axis('y axis', units='px', data=np.linspace(start = 0.0, stop = 512, num = 512), index=1)
        self._dwa = DataRaw('Raw data',
                      data=[np.zeros((x_axis.size,
                                      y_axis.size))],
                      nav_indexes=(0, 1),
                      axes=[x_axis,
                            y_axis,
                            ])
        self._chunk_index = 0
        self._chunk_size = 500000
        self._img = [] # np.zeros((512,512),dtype = np.int64)
        self._current_init_time = 0
        self._init_time_ind = 0
        self._final_time_ind = 0
        self._display_cluster = False

    def setup_docks(self) -> None :
        self.docks['viewer'] = Dock('Viewer2D')
        self.dockarea.addDock(self.docks['viewer'], 'right')
        widget = QtWidgets.QWidget()
        self.viewer2d = Viewer2D(widget)
        self.docks['viewer'].addWidget(widget)

        self.docks['params'] = Dock('Parameters')
        self.dockarea.addDock(self.docks['params'], 'right', self.docks['viewer'])
        self.docks['params'].addWidget(self.settings_tree)
        
        self.docks['info'] = Dock('Info', size=(1, 0.1))
        self.dockarea.addDock(self.docks['info'], 'top', self.docks['params'])
        self.info_label = QtWidgets.QTextEdit("<b>Open a file to start browsing data</b>")
        self.info_label.append("<hr>")

        self.info_label.setReadOnly(True)
        self.info_label.setMaximumHeight(50)
        self.docks['info'].addWidget(self.info_label)

    def setup_actions(self) -> None :
        self.add_action('open_file', 'Open file', 'file_open',
                        tip='Open the data file.',
                        checkable=False,
                        enabled=True,)
        self.add_action('previous', 'Previous data chunk', 'step',
                        tip='Display data of the previous integrated period of time.',
                        checkable=False,
                        enabled=False,
                        flip_h=True,
                        icon_color='green')
        self.add_action('next', 'Next data chunk', 'step',
                        tip='Display data of the next integrated period of time.',
                        checkable=False,
                        enabled=False,
                        icon_color='green')
        self.add_action('cluster_display', 'Display clusters of the data slice', 'target',
                        tip='Display clustering of the current integrated period of time.',
                        checkable=True,
                        enabled=False,
                        icon_color='green')
        
        file_menu = self._menubar.addMenu('File')
        action_menu = self._menubar.addMenu('Data browsing')
        self.affect_to('open_file', file_menu)
        self.affect_to('next', action_menu)
        self.affect_to('previous', action_menu)
        self.affect_to('cluster_display',action_menu)
        
    def connect_things(self):
        self.connect_action('open_file',
                            self.load_data)
        self.connect_action('next',
                            self.show_next)
        self.connect_action('previous',
                            self.show_previous)
        self.connect_action('cluster_display',
                            self.enable_cluster_display)
        
    def enable_cluster_display(self,value) :
        self._display_cluster = value

    def _show_data(self) :
        self._dwa.data = self._img
        self.viewer2d.show_data(self._dwa)

    def show_next(self) :
        self._img = np.zeros((512,512),dtype = np.int64)
        # self._current_init_time += self.duration
        self.forward_time_slice()
        self.build_image()
        # self.display_message(f"Visulasing slice between {self._data_chunk_0[0,self._init_time_ind]} ns and {self._data_chunk_0[0,self._final_time_ind]} ns.")
        self._show_data()

    def show_previous(self) :
        self._img = np.zeros((512,512),dtype = np.int64)
        # self._current_init_time += self.duration
        self.backward_time_slice()
        self.build_image()
        self._show_data()

    def load_data(self) :
        folder_name = QtWidgets.QFileDialog.getOpenFileName(
                None, 'Choose File', os.path.split(str(self.parent))[0],
            filter="Numpy arrays (*.npy)")[0]
        self.settings.child('file_settings','file_path').setValue(folder_name)
        message = f"<b>Loaded file:</b> {folder_name}"
        self.display_message(message)
        self._memmap = np.load(folder_name,mmap_mode='r')
        if not(folder_name == '') :
            self.get_action('next').setEnabled(True)
            self.get_action('previous').setEnabled(True)
            self.get_action('cluster_display').setEnabled(True)
        self.chunk_data()
        # self._current_init_time = self._data_chunk_0[0,0]
        # self.time_slice_data(init_time=self._current_init_time,duration=self.duration)       

    def display_message(self, input_string : str) :
        message = f"[{self._label_count}] : {input_string}"
        self.info_label.append(message)
        scrollbar = self.info_label.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        self._label_count += 1

    def chunk_data(self) :
        upper_0 = min(self._memmap.shape[1],(self._chunk_index+1)*self._chunk_size)
        lower_0 = max(0,self._chunk_index*self._chunk_size)
        upper_1 = min(self._memmap.shape[1],(self._chunk_index+2)*self._chunk_size)
        self._data_chunk_0 = np.asarray(self._memmap[:,lower_0:upper_0],dtype=np.int64)
        self._data_chunk_1 = np.asarray(self._memmap[:,upper_0:upper_1],dtype=np.int64)
        self.display_message(f"Changing data chunks, from index {lower_0} to index {upper_0} and from index {upper_0} to {upper_1}.")

    def forward_time_slice(self) :
        self._init_time_ind = self._final_time_ind
        ind_step = self._init_time_ind + 1
        while self._data_chunk_0[0,ind_step] - self._data_chunk_0[0,self._init_time_ind] < self.duration :
            ind_step += 1
            if ind_step == self._data_chunk_0.shape[1] :
                break
        if ind_step == self._data_chunk_0.shape[1] :
            ind_step = 0
            while self._data_chunk_1[0,ind_step] - self._data_chunk_0[0,-1] < self.duration :
                ind_step += 1
            self._final_time_ind = ind_step
            self._data = np.concatenate((self._data_chunk_0[:,self._init_time_ind:],
                                         self._data_chunk_1[:,:self._final_time_ind]),
                                        axis = 1)
            self._chunk_index +=1
            self.chunk_data()
        else :
            self._final_time_ind = ind_step
            self._data = self._data_chunk_0[:,self._init_time_ind:self._final_time_ind]
            self.display_message(f"Visulasing slice between {self._data_chunk_0[0,self._init_time_ind]} ns and {self._data_chunk_0[0,self._final_time_ind]} ns.")

    def build_image(self) :
        self._img = []
        raw_img = np.zeros((512,512),dtype=np.int64)
        populate_image(raw_img,self._data)
        if self._display_cluster :
            clusts_img = np.zeros((512,512), dtype=np.int64)
            clustered = clusterize_single_chunk(self._data,150)
            populate_clusters(clusts_img,clustered)
            self._img = [clusts_img.copy(),raw_img.copy()]
        else :
            self._img = [raw_img.copy()]

    def backward_time_slice(self) :
        ind_step = self._init_time_ind-1
        while self._data_chunk_0[0,self._init_time_ind-1] - self._data_chunk_0[0,ind_step] < self.duration :
            ind_step -= 1
            if ind_step == 0 : 
                break
        if ind_step == 0 :
            self._chunk_index -=1
            self.chunk_data()
            ind_step = self._data_chunk_0.shape[1] - 1
            while self._data_chunk_1[0,-1] - self._data_chunk_0[0,ind_step] < self.duration :
                ind_step -= 1
            self._final_time_ind = self._init_time_ind
            self._init_time_ind  = ind_step +1
            self._data = np.concatenate((self._data_chunk_0[:,self._init_time_ind:],
                                         self._data_chunk_1[:,:self._final_time_ind]),
                                        axis = 1)
        else :     
            self._final_time_ind = self._init_time_ind
            self._init_time_ind  = ind_step +1
            self.display_message(f"Visulasing slice between {self._data_chunk_0[0,self._init_time_ind]} ns and {self._data_chunk_0[0,self._final_time_ind]} ns.")
            self._data = self._data_chunk_0[:,self._init_time_ind:self._final_time_ind]

    def start(self) :
        raise NotImplementedError("You first need to implement this functionality.")
    
    def stop(self) :
        raise NotImplementedError("You first need to implement this functionality.")
    
    @property
    def duration(self) :
        self._duration = self.settings.child('time_settings','duration').value()
        return self._duration
       
    # def setup_actions(self):
    #     '''
    #     subclass method from ActionManager
    #     '''
    #     logger.debug('setting actions')
    #     self.add_action('quit', 'Quit', 'close2', "Quit program", toolbar=self.toolbar)
    #     self.add_action('grab', 'Grab', 'camera', "Grab from camera", checkable=True, toolbar=self.toolbar)
    #     logger.debug('actions set')

    # def setup_docks(self):
    #     '''
    #     subclass method from CustomApp
    #     '''
    #     logger.debug('setting docks')
    #     self.dock_settings = gutils.Dock('Settings', size=(350, 350))
    #     self.dockarea.addDock(self.dock_settings, 'left')
    #     self.dock_settings.addWidget(self.settings_tree, 10)
    #     logger.debug('docks are set')

    # def connect_things(self):
    #     '''
    #     subclass method from CustomApp
    #     '''
    #     logger.debug('connecting things')
    #     self.actions['quit'].connect(self.quit_function)
    #     self.actions['grab'].connect(self.detector.grab)
    #     logger.debug('connecting done')

    # def setup_menu(self):
    #     '''
    #     subclass method from CustomApp
    #     '''
    #     logger.debug('settings menu')
    #     file_menu = self.mainwindow.menuBar().addMenu('File')
    #     self.affect_to('quit', file_menu)
    #     file_menu.addSeparator()
    #     logger.debug('menu set')

    def value_changed(self, param):
        # logger.debug(f'calling value_changed with param {param.name()}')
        if param.name() == 'base_path':
            if param.value():
                self.settings.child('main_settings', 'something_done').setValue(True)
            else:
                self.settings.child('main_settings', 'something_done').setValue(False)

        # logger.debug(f'Value change applied')

def main():
    app = mkQApp('DataPicker')
    area = DockArea()
    win = QtWidgets.QMainWindow()
    win.setCentralWidget(area)
    data_picker = ClusterViewer(area)

    
    win.show()
    app.exec()

# class ClusterBrowser(ClusterViewer) : 
#     def __init__(self, area):
#         super().__init__(area)

    # def load_data(self,file_path)

if __name__ == '__main__' : 
    main()