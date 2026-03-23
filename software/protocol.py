#coding:utf-8

# * 通讯协议v2.0
# * 2023.12.25
# * 包头： AB 55
# * 包长度：2字节, uint16(np.uint16)
# * 标志符1（1字节）：
#      * bit7-bit4: 设备标识符 0-ads1299(24位ADC), 1-ads1284(31位ADC), 2-ads1263(32位ADC)
#      * bit3-bit0: 电量0-10,表示0~100%
# * 标志符2（1字节）：
#      * bit0:带/不带丢包测试
#      * bit1:带/不带trigger
#      * bit2:带/不带sampleN
#      * bit3:带/不带?
#      * bit4:带/不带?
#      * bit5:带/不带?
#      * bit6:带/不带?
#      * bit7:带/不带?
# * 采样率：2字节, uint16(np.uint16)
# * [1字节丢包测试] 0-255循环    【对应bit0控制的可选字节】
# * [1字节trigger] uint8       【对应bit1控制的可选字节】
# * [可解释字节]                【对应bit2控制的可选字节】
# * [可解释字节]                【对应bit3控制的可选字节】
# * [可解释字节]                【对应bit4控制的可选字节】
# * [可解释字节]                【对应bit5控制的可选字节】
# * [可解释字节]                【对应bit6控制的可选字节】
# * [可解释字节]                【对应bit7控制的可选字节】
# * 数据序列
# * 1字节校验
# * 特别说明：默认一个数据包包含一个采样点。需要包含多个采样点时，可使用可解释字节补充说明

# // * protocol v2.0
# // * 2023.12.25
# // * Head： AB 55
# // * PackageLen：2Bytes, uint16(little endian)
# // * IndicateByte1（1Byte）：
# //      * bit7-bit4: device ID 0-ads1299, 1-ads1284, 2-ads1263
# //      * bit3-bit0: battery power level: 0-10 -> 0~100%
# // * IndicateByte2（1字节）：
# //      * bit0:include/noninclude sampleID
# //      * bit1:include/noninclude trigger
# //      * bit2:include/noninclude sampleN
# //      * bit3:include/noninclude ?
# //      * bit4:include/noninclude ?
# //      * bit5:include/noninclude ?
# //      * bit6:include/noninclude ?
# //      * bit7:include/noninclude ?
# // * Srate：2字节, uint16(little endian)
# // * [1Byte sampleID] 0-255 cycle   【optional depends on bit0】
# // * [1Byte trigger]                【optional depends on bit1】
# // * [1Byte sampleN]               【optional depends on bit2】
# // * [1Byte optional]               【optional depends on bit3】
# // * [1Byte optional]               【optional depends on bit4】
# // * [1Byte optional]               【optional depends on bit5】
# // * [1Byte optional]               【optional depends on bit6】
# // * [1Byte optional]               【optional depends on bit7】
# // * data array
# // * 1Byte check (sum check)
# // * specially: one sample included in one package in default, the optional bytes may
#      indicate how many samples include in one package if needed

# //* 通讯协议v3.0
# //* 2024.11.25
# //* 包头： AB 55
# //* 包长度：2字节, uint16((numpy.uint16))
# //* 标志符1（1字节）：
# //     * bit7-bit4: 设备标识符 0-ads1299(24位ADC), 1-ads1284(31位ADC), 2-ads1263(32位ADC)
# //     * bit3-bit0: 电量0-10,表示0~100%
# //* 标志符2（1字节）：
# //     * bit0~bit1:工作模式。
# //				 ** 0-停机模式，在该模式下下位机每1秒向上位机发送一次数据。发送数据包括：包头、包长度、标志1、标志2、校验。
# //         ** 1-EEG采集模式，该模式下传输的是EEG数据
# //         ** 2-阻抗检测模式，该模式下传输的是阻抗检测的数据，其中数据序列同EEG模式。
# //         ** 3-预留模式
# //     * bit2:
# //         ** 带/不带丢包测试, 如果带，在可解释字节字段添加一个字节,为_sampleID。
# //     * bit3:
# //         ** 带/不带trigger, 如果带，在可解释字节字段增加一个字节,为trigger
# //     * bit4:阻抗检测模式下：bias连接情况
# //     * bit5:阻抗检测模式下：ref连接情况
# //
# //* 采样率：2字节, uint16((numpy.uint16))
# //* [1字节丢包测试] 0-255循环    【对应bit0控制的可选字节】
# //* [1字节trigger] uint8       【对应bit1控制的可选字节】
# //* [可选字节]  对应其他控制位（如果有）
# //* 数据序列
# //* 1字节校验
# //* 特别说明：默认一个数据包包含一个采样点。需要包含多个采样点时，可使用可解释字节补充说明


import numpy as np

class DevData:
    def __init__(self):
        self.devID = 0
        self.devMode = 0
        self.srate = 0
        self.batLevel = 10
        self.pakID = b''
        self.trigger = b''
        self.emgpayload = b''
        self.accBytes = b''
        self.gloveBytes = b''
        self.emgChs = 0
        self.accChs = 0
        self.gloveChs = 0
        self.biasconnect = 0
        self.refconnect = 0
        self.sampleInterval = 0
        self.sampleN = 0
        self.includeID = False
        self.includeTri = False
        self.includeAcc = False
        self.includeGlove = False

    def reset(self):
        self.devID = 0
        self.devMode = 0
        self.srate = 0
        self.batLevel = 10
        self.pakID = b''
        self.trigger = b''
        self.emgpayload = b''
        self.emgChs = 8
        self.accChs = 12
        self.gloveChs = 14
        self.biasconnect = 0
        self.refconnect = 0
        self.sampleInterval = 0
        self.sampleN = 0
        self.includeID = False
        self.includeTri = False
        self.includeAcc = False
        self.includeGlove = False
        self.gloveBytes = b''


class Protocol():
    UUID = 'emg-gloveV2'
    def __init__(self,uuid):
        if self.UUID != uuid:
            raise IOError("protocol match error!")

        self.u16dt = np.dtype('uint16')
        self.devData = DevData()
        self.buffer = b''
        self.payloadStartIndex = 0

        self.bytesOnePoint = [3,4,4,4,4] # 0-ads1299,3字节,1-ads1284,4字节,2-ads1263,4字节
        self.packLen = 0
        self.package = b''

        self.accBytesWan = b'\x00'*12
        self.accBytesHand = b'\x00'*12
        self.gloveBytes = b'\x00'*28

    def loadBuffer(self,buf): # 加载buffer
        self.buffer = buf

    def headVerify(self):
        return self.buffer[0] == 0xAB and self.buffer[1] == 0x55

    def paklenVerify(self):
        self.pakLen = int(np.frombuffer(self.buffer[2:4], dtype=self.u16dt)[0])
        return len(self.buffer) >= self.pakLen, self.pakLen

    def getEpochAndVerify(self):
        self.package = self.buffer[:self.pakLen]
        return (sum(self.package[:-1]) & 0xff) == self.package[-1]

    def parsePak(self):
        self.devData.reset()

        identifier1 = self.package[4] # 标志字节1
        self.devData.devID = identifier1 >> 4
        self.devData.batLevel = identifier1 & 0x0f

        self.devData.srate = np.frombuffer(self.package[6:8], dtype=self.u16dt)[0]

        identifier2 = self.package[5]
        ind = 8
        self.devData.includeID = identifier2 & 0x01
        if self.devData.includeID:
            self.devData.pakID = self.package[ind:ind + 1]
            ind += 1

        self.devData.includeTri = (identifier2 >> 1) & 0x01
        if self.devData.includeTri:
            self.devData.trigger = self.package[ind:ind + 1]
            ind += 1

        if (identifier2 >> 2) & 0x01:
            sampleN = self.package[ind]
            ind += 1
        else:
            sampleN = 1

        includeAcc = (identifier2 >> 3) & 0x01
        if includeAcc:
            self.accBytesWan = self.package[ind:ind + 12]
            ind += 12

        includeGlove = (identifier2 >> 4) & 0x01
        if includeGlove:
            self.gloveBytes = self.package[ind:ind + 28]
            self.accBytesHand = self.package[ind+28:ind+40]

            # data = np.frombuffer(self.devData.gloveBytes, dtype=np.uint16) * 0.0008
            # print(data)
            # # print(data.size)
            ind += 40

        # self.devData.optionBytes = b''
        # if (identifier2 >> 5) & 0x01:
        #     self.devData.optionBytes += self.package[ind:ind + 1]
        #     ind += 1
        #
        # if (identifier2 >> 6) & 0x01:
        #     self.devData.optionBytes += self.package[ind:ind + 1]
        #     ind += 1
        #
        # if (identifier2 >> 7) & 0x01:
        #     self.devData.optionBytes += self.package[ind:ind + 1]
        #     ind += 1

        self.devData.accBytes = self.accBytesWan + self.accBytesHand
        self.devData.gloveBytes = self.gloveBytes
        self.devData.includeAcc = True
        self.devData.includeGlove = True
        self.devData.emgpayload = self.package[ind:-1]
        self.devData.sampleN = sampleN
        return self.devData