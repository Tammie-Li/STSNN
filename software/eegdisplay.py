#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Time    : 2024/11/28 9:28
Author  : mrtang
Email   : 810899799@qq.com
"""

import pyqtgraph as pg
pg.setConfigOption('background', (255, 255, 255))
from datamanager import DataManager
from shm2 import CreateShm,EEGTYPE
from butterfilter import ButterFilter
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget
import numpy as np

import json


COLORS = [(255,0,0),(0,0,255),(0,0,0),(255,0,255)]
yscale = [20,50,100,200,500,2,10,100,1,5,0]
#         uv,uv,uv,  uv,uv, mv,mv,mv,v,v,?
ygain =  [1,1,1,1,1,1e-3,1e-3,1e-3,1e-6,1e-6,1]

'''负责绘图
绘图fs: 250
绘图进程从共享内存区读取数据，获得设备的相关信息，进行绘图曲线初始化
后，对实时数据进行绘制 '''

class EEGDisplay(QWidget):
    psig = pyqtSignal(str)
    def __init__(self,parentUI,config):
        super(EEGDisplay, self).__init__()
        self.shm = CreateShm(master = False)

        # 控制电量更新频率
        self.batC = 80 # 大约4秒更新一次
        self.batUC = 0

        # 绘图相关
        self.flttype = 2  # 0-None, 1-high, 2-band
        self.scale = 10
        self.ygain = 1
        self.config = config
        rawsrate = config['srate']
        self.downsampleScale = rawsrate // 250
        self.localSrate = rawsrate // self.downsampleScale
        self.emgChsNum = len(self.config['emgchs'])
        self.accChsNum = self.config['accChsNum']
        self.gloveChsNum = self.config['gloveChsNum']
        self.totalChsNum = self.emgChsNum + self.accChsNum + self.gloveChsNum
        self.curves = []    # 绘图曲线组
        self.period = 4     # 绘图时长
        self.prepare = True  # True： update_one_frame跳过

        self.cccc = 0

        # # 绘图数据配置
        self.dm = DataManager()  # 绘图数据管理器
        self.accdm = DataManager()  # 绘图数据管理器
        self.ui = parentUI
        self.ui.yrange_cmb.currentIndexChanged.connect(self.relayout)   # y尺度调整

        self.psig.connect(self.updateBar)

        # 绘图控制
        self.pgplot = pg.PlotWidget()
        self.pgplot.showGrid(True,True)

        self.handaccplot = pg.PlotWidget()
        self.handaccplot.showGrid(True, True)

        self.wanaccplot = pg.PlotWidget()
        self.wanaccplot.showGrid(True, True)

        # 初始化绘图
        self.virgin = True
        self.relayout()

        # 数据读取和滤波控制
        self.index = 0
        self.filter = ButterFilter()
        self.filter.reset(srate=self.localSrate, chs=self.emgChsNum,
                           fltparam=[(49, 51), (5, 90), (1, 0), None],eegtype=EEGTYPE)
        # self.ui.emgLayout.addWidget(self.pgplot)
        self.pgTimer = pg.QtCore.QTimer()
        self.pgTimer.timeout.connect(self.update_one_frame)

    def updateBar(self,mess):
        try:
            dic = json.loads(mess)
            data = [int(0.8*item) for item in dic['glove']]
            bat = int(dic['batlevel'])
        except:
            return

        self.ui.blendBar0.setValue(data[4])
        self.ui.blendBar1.setValue(data[3])
        self.ui.blendBar2.setValue(data[2])
        self.ui.blendBar3.setValue(data[1])
        self.ui.blendBar4.setValue(data[0])
        self.ui.pressBar0.setValue(5*(43-data[10]))
        self.ui.pressBar1.setValue(5*(43-data[9]))
        self.ui.pressBar2.setValue(5*(43-data[8]))
        self.ui.pressBar3.setValue(5*(43-data[7]))
        self.ui.pressBar4.setValue(5*(43-data[6]))

        # print(data[6:])

        self.ui.batLevel.setValue(bat)

    def startPloting(self,flg):
        if flg:   self.pgTimer.start(5)
        else:     self.pgTimer.stop()

    def addToMainWin(self,flg):
        if flg:
            self.ui.emgLayout.addWidget(self.pgplot)
            self.ui.handAccLayout.addWidget(self.handaccplot)
            self.ui.armAccLayout.addWidget(self.wanaccplot)
            self.pgplot.show()
            self.handaccplot.show()
            self.wanaccplot.show()
        else:
            self.pgplot.hide()
            self.handaccplot.hide()
            self.wanaccplot.hide()
            self.ui.emgLayout.removeWidget(self.pgplot)
            self.ui.handAccLayout.removeWidget(self.handaccplot)
            self.ui.armAccLayout.removeWidget(self.wanaccplot)

    def relayout(self):
        self.prepare = True  #暂时屏蔽绘图更新
        scale = yscale[self.ui.yrange_cmb.currentIndex()]   # 当前y方向上的尺度
        self.ygain = ygain[self.ui.yrange_cmb.currentIndex()]
        self.pgplot.setYRange(0,scale*self.emgChsNum)

        if self.virgin:
            self.virgin = False
            self.curves = []
            for idx, ch in enumerate(self.config['emgchs']):
                curve = pg.PlotCurveItem(pen=pg.mkPen(color=(250,100,50), width=1))
                self.pgplot.addItem(curve)
                curve.setPos(0, idx * scale + 0.5 * scale)
                self.curves.append(curve)

            self.dm.config(self.localSrate, self.emgChsNum, self.period, EEGTYPE)
            self.pgplot.setXRange(0, self.localSrate * self.period)

            self.handacccurves = []
            self.wanacccurves = []
            colors = [(255,0,0),(0,255,0),(0,0,255),(0,0,0),(0,0,0),(0,0,0)]
            for i in range(6):
                curve = pg.PlotCurveItem(pen=pg.mkPen(color=colors[i], width=1))
                self.handaccplot.addItem(curve)
                curve.setPos(0, 0)
                self.handacccurves.append(curve)

            for i in range(6):
                curve = pg.PlotCurveItem(pen=pg.mkPen(color=colors[i], width=1))
                self.wanaccplot.addItem(curve)
                curve.setPos(0, 0)
                self.wanacccurves.append(curve)

            self.accdm.config(self.localSrate, self.accChsNum, self.period, EEGTYPE)
            self.handaccplot.setXRange(0, self.localSrate * self.period)
            self.wanaccplot.setXRange(0, self.localSrate * self.period)


        if scale != self.scale:
            self.scale = scale
            for idx, ch in enumerate(self.config['emgchs']):
                self.curves[idx].setPos(0, idx * scale + 0.5 * scale)

        self.prepare = False

    def update_one_frame(self):
        # 读数据
        ind = self.shm.info[0]
        if ind == 0:    return  # 设备未启动
        if self.prepare:    return      # 准备状态不更新

        if ind != self.index:  # eeg数据有更新
            self.index = ind
            if self.shm.getvalue('mode') != 1: # 确保在正确的模式下
                return

            self.shm.setvalue('plotting', 1)

            curdataindx = int(self.shm.getvalue('curdataindex'))
            pp = int(curdataindx/self.totalChsNum)

            dat = self.shm.eeg[:curdataindx].reshape(pp, self.totalChsNum).transpose()
            self.shm.setvalue('curbyteindex', 0)
            self.shm.setvalue('curdataindex', 0)
            self.shm.setvalue('plotting', 0)

            eeg = dat[:self.emgChsNum, ::self.downsampleScale]

            self.filter.update(eeg)  # 滤波
            if self.flttype == 0:  # none
                self.dm.update(self.filter.rawdata)
            elif self.flttype == 1:  # high pass
                self.dm.update(self.filter.hdata)
            elif self.flttype == 2:  # band pass
                self.dm.update(self.filter.bdata)
            else:
                self.dm.update(self.filter.rawdata)

            if self.dm.data is None:    return
            for id in range(self.emgChsNum):
                self.curves[id].setData(self.dm.data[id, :] * self.ygain)

            accdd = dat[self.emgChsNum:self.emgChsNum + self.accChsNum, ::self.downsampleScale]
            self.accdm.update(accdd)
            for i in range(6):
                self.wanacccurves[i].setData(self.accdm.data[i,:])
                self.handacccurves[i].setData(self.accdm.data[i+6,:])


            self.cccc += 1
            self.cccc %= 4   # every 0.2s
            if self.cccc == 0:
                glovedd = dat[self.emgChsNum + self.accChsNum:,-1]
                dic = {"glove":glovedd.tolist(),"batlevel":int(self.shm.getvalue("batlevel"))}
                mess = json.dumps(dic)
                self.psig.emit(mess)


    def release(self):
        self.shm.release()