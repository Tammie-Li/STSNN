import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import pygame
from pygame.locals import *
import sys, os

# 初始化 Pygame
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Real-Time Hand Pose Visualization")
clock = pygame.time.Clock()



class ReadGmData():
    verison = 2
    def __init__(self, path):
        self.path = path

    def readfile(self):
        buffer = b''
        with open(self.path,'rb') as f:
            buffer = f.read()

        headlen = np.frombuffer(buffer[:4],dtype=np.int32)[0]
        headay = np.frombuffer(buffer[:4*headlen],dtype=np.int32)

        if headay[1] != self.verison:
            raise IOError('file verison dismatch!')

        self.srate = headay[3]
        dt = headay[2]
        if dt == 1:
            raise IOError('evt file is currently upsupported')

        if dt == 2:
            self.adctype = np.dtype(np.float32)
        elif dt == 3:
            self.adctype = np.dtype(np.float64)
        else:
            raise IOError('unknow adc data type')

        self.emgChs = headay[4]
        self.accChs = headay[5]
        self.gloveChs = headay[6]
        self.totalChs =  self.emgChs + self.accChs + self.gloveChs + 1

        dataBuffer = buffer[4*headlen:]
        L = int((len(dataBuffer)//(self.totalChs*self.adctype.itemsize))*(self.totalChs*self.adctype.itemsize))
        sampleN = L//(self.totalChs*self.adctype.itemsize)
        dataBuffer = dataBuffer[:L]
        adcData = np.frombuffer(dataBuffer,dtype=self.adctype)
        data = adcData.reshape(sampleN,self.totalChs).transpose()
        return {'srate':self.srate,'emgchs':self.emgChs,'accchs':self.accChs,'glovechs':self.gloveChs,'data':data}

# 定义手的模型（详细的手指和手背）
hand_model = {
    "palm": [  # 手背（四边形）
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)
    ],
    "thumb": [  # 拇指（3段）
        (0.2, 0, 0), (0.5, -0.3, 0), (0.8, -0.5, 0), (1.0, -0.7, 0)
    ],
    "index": [  # 食指（3段）
        (0.8, 0, 0), (1.0, 0.3, 0), (1.2, 0.6, 0), (1.4, 0.9, 0)
    ],
    "middle": [  # 中指（3段）
        (0.5, 0, 0), (0.5, 0.5, 0), (0.5, 1.0, 0), (0.5, 1.5, 0)
    ],
    "ring": [  # 无名指（3段）
        (0.3, 0, 0), (0.2, 0.5, 0), (0.1, 1.0, 0), (0.0, 1.5, 0)
    ],
    "pinky": [  # 小指（3段）
        (0.0, 0, 0), (-0.2, 0.3, 0), (-0.4, 0.6, 0), (-0.6, 0.9, 0)
    ]
}

# 定义颜色
colors = {
    "palm": (255, 200, 150),  # 手背颜色
    "thumb": (255, 0, 0),     # 拇指颜色
    "index": (0, 255, 0),     # 食指颜色
    "middle": (0, 0, 255),    # 中指颜色
    "ring": (255, 255, 0),    # 无名指颜色
    "pinky": (255, 0, 255)    # 小指颜色
}

# 定义 IMU 数据模拟器（加速度计和陀螺仪）
class IMUSimulator:
    def __init__(self):
        file_reader = ReadGmData(os.path.join(os.getcwd(), "Data", "left.dat"))
        self.data = file_reader.readfile()["data"][14:20, :].T
        self.acc = self.data[0, :3]
        self.gyro = self.data[0, 3:]

        self.idx = 1

    def update(self, dt):
        # 模拟加速度和角速度变化
        self.acc = np.array([self.data[self.idx, 1], self.data[self.idx, 0], self.data[self.idx, 2]])
        self.gyro = np.array([self.data[self.idx, 4], self.data[self.idx, 3], self.data[self.idx, 5]])

        print(self.acc)
        self.idx += 1
        return self.acc, self.gyro

# 姿态解算（使用互补滤波）
class PoseEstimator:
    def __init__(self):
        self.quaternion = np.array([1, 0, 0, 0])  # 初始姿态（四元数）
        self.dt = 0.01

    def update(self, acc, gyro):
        # 互补滤波参数
        alpha = 0.98

        # 加速度计姿态估计（俯仰和横滚）
        acc_norm = acc / np.linalg.norm(acc)
        roll = np.arctan2(acc_norm[1], acc_norm[2])
        pitch = np.arctan2(-acc_norm[0], np.sqrt(acc_norm[1]**2 + acc_norm[2]**2))
        acc_quat = R.from_euler('xyz', [roll, pitch, 0]).as_quat()

        # 陀螺仪姿态估计
        gyro_quat = R.from_rotvec(gyro * self.dt).as_quat()
        self.quaternion = alpha * (R.from_quat(self.quaternion) * R.from_quat(gyro_quat)).as_quat() + (1 - alpha) * acc_quat
        return self.quaternion

# 绘制手的模型
def draw_hand(screen, hand_model, pose):
    # 获取旋转矩阵
    # 获取旋转矩阵
    rotation_matrix = R.from_quat(pose).as_matrix()

    # 缩放因子和偏移量
    scale = 100
    offset = np.array([WIDTH / 2, HEIGHT / 2, 0])

    # 绘制手背和手指
    for part, points in hand_model.items():
        # 将点转换为屏幕坐标
        screen_points = []
        for point in points:
            point_rotated = np.dot(rotation_matrix, np.array(point) * scale) + offset
            screen_points.append((int(point_rotated[0]), int(point_rotated[1])))

        # 绘制线段
        if part == "palm":  # 手背（四边形）
            pygame.draw.polygon(screen, colors[part], screen_points, 2)
        else:  # 手指（线段）
            for i in range(len(screen_points) - 1):
                pygame.draw.line(screen, colors[part], screen_points[i], screen_points[i + 1], 2)

# 主程序
def main():
    imu_simulator = IMUSimulator()
    pose_estimator = PoseEstimator()

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

        # 更新 IMU 数据
        acc, gyro = imu_simulator.update(pose_estimator.dt)

        # 更新姿态
        pose = pose_estimator.update(acc, gyro)

        # 绘制手的模型
        screen.fill((0, 0, 0))
        draw_hand(screen, hand_model, pose)
        pygame.display.flip()

        # 控制帧率
        clock.tick(500)

if __name__ == "__main__":
    main()