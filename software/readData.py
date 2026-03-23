
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

datapath = r'./data.dat'

class ReadGmData():
    verison = 2
    def __init__(self,path = datapath):
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

if __name__ == '__main__':
    rd = ReadGmData(r'./data.dat')
    rd.readfile()