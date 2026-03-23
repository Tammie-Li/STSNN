#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Time    : 2024/11/28 10:16
Author  : mrtang
Email   : 810899799@qq.com
"""

import numpy as np
from multiprocessing import Event, Queue
import time
import math
from protocol import Protocol
from shm2 import EEGTYPE,CreateShm,EEGMAXLEN

devconfig = {   'vref':4.5,
                'bits':24,
                'gain':[24,24,24,24,24,24,24,24],
                'accrang':8,
                'gyrrang':1024}

class DataDecoder():
    def __init__(self,uuid):
        self.shm = CreateShm(master = False)
        self.decoder24 = ADC24Decoder()
        self.decoderQmi = QmiDecoder()
        self.decoderGlove = GloveDecoder()
        self.protocol = Protocol(uuid)
        self.mode = 1
        self.sampleCount = 0
        self.payloads = b''
        self.accBytes = b''
        self.gloveBytes = b''
        self.buffer = b''
        self.ids = b''
        self.tris = b''
        self.triggerbytes = b''

        self.batLevel = 0
        self.saveFlg = 0
        self.file = None
        self.timestamps = []
        self.triggers = []
        self.curr_trigger = 0


        if EEGTYPE == 'float32':
            self.typeLen = 4
        elif EEGTYPE == 'float64':
            self.typeLen = 8
    
    def update_trigger(self, trigger):
        self.curr_trigger = trigger

    def parseData(self, buffer,*args):
        stamp = args[0]
        self.buffer += buffer
        Len = len(self.buffer)
        indx = 0
        while indx < Len-7:
            self.protocol.loadBuffer(self.buffer[indx:])
            if self.protocol.headVerify(): # 头部校验成功
                includePak, pakLen = self.protocol.paklenVerify()  # 校验包长度
                if includePak:  # 长度足够容纳一个数据包
                    if self.protocol.getEpochAndVerify():  # 截取数据包并校验
                        devData = self.protocol.parsePak()
                        indx += pakLen
                        # 给当前数据包分配时间戳
                        # faraway = math.ceil((Len - indx) / pakLen)  # 倒数第几个包
                        # st = stamp - faraway * devData.sampleInterval # 对齐的时间戳
                        # self.collectAll(devData,st)
                        self.collectAll(devData, stamp)
                    else:
                        indx += 1
                else: #长度不够，跳出，下次再来
                    break
            else: #继续向后寻找
                indx += 1

        self.buffer = self.buffer[indx:]
        self.dataarange()  # 整理数据

    def collectAll(self,dat,stamp):  # 拿到一个新包，进行解析
        self.batLevel = dat.batLevel
        self.devID = dat.devID
        self.payloads += dat.emgpayload
        self.accBytes += dat.accBytes
        self.gloveBytes += dat.gloveBytes
        self.triggerbytes += dat.trigger

        self.ids += dat.pakID
        self.tris += dat.trigger
        self.emgChs = dat.emgChs
        self.accChs = dat.accChs
        self.gloveChs = dat.gloveChs
        self.srate = dat.srate
        self.sampleCount += dat.sampleN
        self.biasconnect = dat.biasconnect
        self.refconnect = dat.refconnect
        self.timestamps.append(stamp)
        self.triggers.append(self.curr_trigger)




    def dataarange(self): # 调用者已经控制了节奏
        self.shm.setvalue('batlevel', self.batLevel)
        self.shm.setvalue('mode',self.mode)
        # # print(self.mode)
        # if self.mode == 0:
        #     return

        while self.shm.getvalue('plotting'):
            time.sleep(0.001)

        if len(self.payloads) == 0:
            return

        # n×8,8-8通道
        emgdataay = self.decoder24.decode(self.payloads, self.sampleCount, self.emgChs)
        # n×12,6（主机腕部的imu芯片，accx,accy,accz,gyrx,gyry,gryz）+ 6（手套上的imu芯片，accx,accy,accz,gyrx,gyry,gryz）
        accdataay = self.decoderQmi.decode(self.accBytes, self.sampleCount, self.accChs)
        # n×14,0-4:压力传感器依次对应小拇指，无名指，中指，食指，大拇指，6-10：弯曲传感器依次对应小拇指，无名指，中指，食指，大拇指
        glovedataay = self.decoderGlove.decode(self.gloveBytes, self.sampleCount, self.gloveChs)
        # 时间戳
        sts = np.array(self.timestamps,dtype=np.float64)
        sts = sts[:,np.newaxis]
        alldataRwtStamp = np.hstack((emgdataay,accdataay,glovedataay[:,0:5],glovedataay[:,6:11],sts))


        alldataRwt = np.hstack((emgdataay,accdataay,glovedataay))

        print(alldataRwt.shape)

        alldataFlatten = alldataRwt.flatten()
        L = alldataFlatten.size

        self.shm.setvalue('emgchs',self.emgChs)
        self.shm.setvalue('accchs',self.accChs)
        self.shm.setvalue('glovechs',self.gloveChs)
        self.shm.setvalue('srate',self.srate)

        curdataindex = self.shm.getvalue('curdataindex')
        if curdataindex + L > EEGMAXLEN:
            curdataindex = 0

        self.shm.eeg[curdataindex:curdataindex+L] = alldataFlatten[:]
        self.shm.setvalue('curdataindex', curdataindex + L)
        self.shm.setvalue('curbyteindex', (curdataindex + L)*self.typeLen)
        self.shm.info[0] += 1

        # 保存数据相关 start ====================================================
        if self.saveFlg == 0:
            if self.shm.getvalue('savedata') == 1:  # 开启保存
                pth = self.shm.getPath()
                # 写二进制文件
                # 写入头信息(头信息序列使用int32写入)
                # headlen, 头信息长度,7
                # fileVersion，文件格式, 2 (v2.0)
                # datatype, 数据格式：1- evt数据（int32）, 2-adc数据（float32）, 3-adc数据（float64）
                # srate,采样率：500，
                # emgchs,
                # accchs,
                # glovechs

                try:
                    self.file.close()
                except:
                    pass

                self.file = open(pth, 'wb')
                if self.typeLen == 4:
                    ay = np.array([7,2,2, self.srate, self.emgChs, self.accChs, self.gloveChs], dtype=np.int32)
                else:
                    ay = np.array([7,2,3, self.srate, self.emgChs, self.accChs, self.gloveChs], dtype=np.int32)

                self.file.write(ay.tobytes())  # 头信息
                self.saveFlg = 1

        else:  # self.saveFlg == 1:
            if self.shm.getvalue('savedata') == 0:  # 结束保存
                self.file.close()
                self.saveFlg = 0

            else:  # 正常保存
                stamp = np.array([self.timestamps]).transpose()
                triggers = np.array([self.triggers]).transpose()
                ay = np.hstack((alldataRwt,stamp, triggers)).astype(EEGTYPE).flatten()

                # print(np.hstack((alldataRwt,stamp, triggers)).shape, np.hstack((alldataRwt,stamp, triggers))[:, -1])

                self.file.write(ay.tobytes())
                self.saveFlg = 1

        # 保存数据相关 end ===================================================
        self.sampleCount = 0
        self.payloads = b''
        self.accBytes = b''
        self.gloveBytes = b''
        self.triggerbytes = b''

        
        self.timestamps = []
        self.triggers = []

        self.ids = b''
        self.tris = b''


class ADC24Decoder():
    def __init__(self,config = devconfig):
        self.rawdt = np.dtype('int32')
        self.rawdt = self.rawdt.newbyteorder('>')
        vref = config['vref']
        bits = config['bits']
        gain = config['gain']
        self.facs = np.array([self._calFac(vref,bits,g) for g in gain])
        self.facs = self.facs[np.newaxis,:]  # 组织为二维数组

    def _calFac(self,vref,bits,gain):
        return vref / (gain*(2**bits - 1)) * 1e6

    def getchs(self,payloads,N): # payloads必须是一个采样数据包的
        return int(len(payloads)/3/N)

    def _tobuf32(self, buf24):
        if buf24[0] > 127:
            return b'\xff' + buf24[:3]
        else:
            return b'\x00' + buf24[:3]

    def decode(self,payloads,sampleN,chs):
        tmbuf = [self._tobuf32(payloads[i:i + 3]) for i in range(0, len(payloads), 3)]
        buf = b''.join(tmbuf)
        eeg = np.frombuffer(buf, dtype=self.rawdt).astype(EEGTYPE).reshape(sampleN,chs) #组织为列向量
        fac = np.repeat(self.facs,sampleN,axis=0)
        eeg = eeg * fac
        # return eeg.flatten()
        return eeg


class QmiDecoder():
    '''
    acc:
    +-2g:  16384
    +-4g:  8192
    +-8g:  4096
    +-16g: 2048
    gyr:
    +-16dps:   2048
    +-32dps:   1024
    +-64dps:   512
    +-128dps:  256
    +-256dps:  128
    +-512dps:  64
    +-1024dps: 32
    +-2048dps: 16
    '''

    def __init__(self,config = devconfig):
        self.rawdt = np.dtype('int16')
        accrang = config['accrang']     #2,4,8,16
        accfac = 9.8*2*accrang/65536
        gyrrang = config['gyrrang']  #16,32,64,128,256,512,1024,2048
        gyrfac = 2*gyrrang/65536
        self.facs = np.array([accfac]*3 + [gyrfac]*3 + [accfac]*3 + [gyrfac]*3)
        self.facs = self.facs[np.newaxis,:]  # 组织为二维数组

    def decode(self,payloads,sampleN,chs):
        data = np.frombuffer(payloads, dtype=self.rawdt).astype(EEGTYPE).reshape(sampleN,chs) #组织为列向量
        fac = np.repeat(self.facs,sampleN,axis=0)
        data = data * fac
        # return data.flatten()
        return data

class GloveDecoder():
    def __init__(self,config = devconfig):
        self.rawdt = np.dtype('uint16')
        self.fac = 0.025

    def decode(self,payloads,sampleN,chs):
        data = np.frombuffer(payloads, dtype=self.rawdt).astype(EEGTYPE).reshape(sampleN,chs) #组织为列向量
        data = data * self.fac
        # return data.flatten()
        return data