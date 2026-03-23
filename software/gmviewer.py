#coding:utf-8

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QApplication
from ui.gmviewerui import Ui_MainWindow
from ui.paradigmui import Ui_Paradigm
from devmanager import devManager
from shm2 import CreateShm
from PyQt5.QtCore import pyqtSignal

from eegdisplay import EEGDisplay

import json
from myMessageBox import showMessageBox

'''主程序入口'''

class MouseController(QtWidgets.QWidget):
    def __init__(self):
        super(MouseController, self).__init__()
        self.initUI()

        # 初始化红色小圆点的状态
        self.red_dot_visible = False
        self.red_dot_position = QtCore.QPoint(0, 0)

        # 设置鼠标跟踪
        self.setMouseTracking(True)

        # 回调函数列表
        self.move_callbacks = []
        self.click_callbacks = []

    def initUI(self):
        """初始化窗口"""
        self.setWindowTitle("鼠标控制 - 外部控制器")
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("background-color: white;")  # 设置背景颜色

    def paintEvent(self, event):
        """重绘事件，用于绘制红色小圆点"""
        painter = QtGui.QPainter(self)
        if self.red_dot_visible:
            painter.setBrush(QtGui.QColor(255, 0, 0))  # 红色
            painter.setPen(QtGui.QColor(255, 0, 0))  # 红色
            painter.drawEllipse(self.red_dot_position, 5, 5)  # 绘制小圆点

    def register_move_callback(self, callback):
        """注册鼠标移动回调函数"""
        self.move_callbacks.append(callback)

    def register_click_callback(self, callback):
        """注册鼠标点击回调函数"""
        self.click_callbacks.append(callback)

    def move_mouse(self, dx, dy):
        """移动鼠标"""
        # 更新小圆点位置
        new_x = self.red_dot_position.x() + dx
        new_y = self.red_dot_position.y() + dy
        self.red_dot_position = QtCore.QPoint(new_x, new_y)
        self.red_dot_visible = True
        self.update()

        # 触发回调函数
        for callback in self.move_callbacks:
            callback(self.red_dot_position)

    def click_mouse(self):
        """模拟鼠标点击"""
        self.red_dot_visible = False
        self.update()

        # 触发回调函数
        for callback in self.click_callbacks:
            callback(self.red_dot_position)

    def mouseMoveEvent(self, event):
        """鼠标移动事件（用于测试）"""
        self.red_dot_position = event.pos()
        self.red_dot_visible = True
        self.update()

    def mousePressEvent(self, event):
        """鼠标点击事件（用于测试）"""
        self.red_dot_visible = False
        self.update()

# 外部控制器示例
class ExternalController:
    def __init__(self, mouse_controller):
        self.mouse_controller = mouse_controller

    def move_left(self):
        """向左移动鼠标"""
        self.mouse_controller.move_mouse(-10, 0)

    def move_right(self):
        """向右移动鼠标"""
        self.mouse_controller.move_mouse(10, 0)

    def move_up(self):
        """向上移动鼠标"""
        self.mouse_controller.move_mouse(0, -10)

    def move_down(self):
        """向下移动鼠标"""
        self.mouse_controller.move_mouse(0, 10)

    def click(self):
        """模拟点击"""
        self.mouse_controller.click_mouse()



class gmViewer(QtWidgets.QMainWindow):
    mainsig = pyqtSignal(str)
    _sigal_send_trigger = pyqtSignal(int)
    def __init__(self,configpath = './/config.js'):
        super(gmViewer,self).__init__()

        config,err = self.loadConfigs(configpath)
        if config is None:
            showMessageBox("提示",err)
            # sys.exit()

        self.setWindowTitle("基于表面肌电信号的有/无动作手势识别验证平台（软件部分）")

        # 使用StackedWidget实现多个页面
        self.stack_widget = QtWidgets.QStackedWidget(self)

        # 页面1：显示窗口
        plot_widget = QtWidgets.QWidget()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(plot_widget)  #设置plot_widget的UI

        # 页面2：用于构建范式
        paradigm_widget = QtWidgets.QWidget()
        self.ui_para = Ui_Paradigm()
        # self.ui_para = ParadigmUi()
        self.ui_para.setupUi(paradigm_widget)


        # 在QStackedWidget中添加元素
        self.stack_widget.addWidget(plot_widget)      # 添加 plot_widget
        self.stack_widget.addWidget(paradigm_widget)  # 添加 paradigm_widget

        # 设置 QStackedWidget 的大小策略
        self.stack_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding
        )
        # 设置 QStackedWidget 为 MainWindow 的中心部件
        self.setCentralWidget(self.stack_widget)

        # 设置默认显示的页面
        self.current_page_index = 0
        self.stack_widget.setCurrentIndex(self.current_page_index)

        if self.stack_widget.layout() is None:
            layout = QtWidgets.QVBoxLayout(self.stack_widget)
            self.stack_widget.setLayout(layout)


        self._screenResize()
        self.shm = CreateShm(master=True)
        self.devMgr = devManager(self.ui,self.mainsig,config)
        self.mainsig.connect(self.relayout)

        self.eegDis = EEGDisplay(self.ui,config)
        # self.impDis = ImpDisplay(self.ui,config)

        # self.devMgr._signal_trigger.connect(self.link_emg_device_trigger)

        self._sigal_send_trigger.connect(self.devMgr.RDA.dec.update_trigger)


        self.relayout('stop')

    # def link_emg_device_trigger(self):
    #     self._sigal_send_trigger.connect(self.devMgr.RDA.dec.update_trigger)

    def loadConfigs(self,path):
        try:
            with open(path,'r') as f:
                buf = f.read()
        except:
            return None,"配置文件丢失！"

        try:
            config = json.loads(buf)
        except:
            return None,"配置文件不符合json规范！"

        if 'emgchs' not in config or 'srate' not in config:
            return None, "eegchs或srate参数缺失!"
        else:
            if config['emgchs']==[] or config['srate'] not in [250,500,1000]:
                return None,"参数非法！"

        return config,''


    def relayout(self,mess):
        if mess == 'acquireEEG':
            # self.impDis.addToMainWin(False)
            # self.impDis.startPloting(False)
            self.eegDis.addToMainWin(True)
            self.eegDis.startPloting(True)
        elif mess == 'impedanceDetect':
            self.eegDis.addToMainWin(False)
            self.eegDis.startPloting(False)
            # self.impDis.addToMainWin(True)
            # self.impDis.startPloting(True)
        else:
            self.eegDis.addToMainWin(False)
            self.eegDis.startPloting(False)
            # self.impDis.addToMainWin(False)
            # self.impDis.startPloting(False)

    def closeEvent(self, event):
        self.devMgr.release()
        self.shm.release()

    # 由initialize调用
    # 依据显示屏尺寸来初始化界面大小
    def _screenResize(self):
        # 获取显示器相关信息
        desktop = QApplication.desktop()
        # 默认在主显示屏显示
        screen_rect = desktop.screenGeometry(0)
        self.ww = screen_rect.width()
        self.hh = screen_rect.height()
        self.w = int(self.ww*0.92)
        self.h = int(self.hh*0.8)
        self.setGeometry(int((self.ww - self.w)/2), int((self.hh - self.h)/2), self.w, self.h)

    def keyPressEvent(self, event):
        # 自由录数据模式
        if event.key() == QtCore.Qt.Key_F12:  # 按下 F12 键, 切换页面
            self.current_page_index = (self.current_page_index + 1) % self.stack_widget.count()
            self.stack_widget.setCurrentIndex(self.current_page_index)
        elif event.key() == QtCore.Qt.Key_Escape:  # 按下 Escape 键, 休息Trigger
            self._sigal_send_trigger.emit(0)

        elif event.key() == QtCore.Qt.Key_F1:  # 按下 "1" 键
            self._sigal_send_trigger.emit(1)

        elif event.key() == QtCore.Qt.Key_F2:  # 按下 "2" 键
            self._sigal_send_trigger.emit(2)

        elif event.key() == QtCore.Qt.Key_F3:  # 按下 "3" 键
            self._sigal_send_trigger.emit(3)

        elif event.key() == QtCore.Qt.Key_F4:  # 按下 "4" 键
            self._sigal_send_trigger.emit(4)

        elif event.key() == QtCore.Qt.Key_F5:  # 按下 "5" 键
            self._sigal_send_trigger.emit(5)

        elif event.key() == QtCore.Qt.Key_F6:  # 按下 "6" 键
            self._sigal_send_trigger.emit(6)

        super().keyPressEvent(event)
 
if __name__ == '__main__':
    import sys
    # import multiprocessing
    # multiprocessing.freeze_support()
    app = QApplication(sys.argv)
    a = gmViewer()
    a.show()
    sys.exit(app.exec_())