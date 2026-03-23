import numpy as np
from scipy.stats import entropy
from scipy.signal import lfilter, welch
from scipy.linalg import hankel
EEGTYPE = 'float64'
import matplotlib.pyplot as plt
import os, struct
from scipy import signal
from scipy.ndimage import median_filter



class ButterFilter:
    def __init__(self):
        pass

    def reset(self,srate = 500, chs = 8, fltparam = [(49,51),(0.5,45),(1,0),None],eegtype='float32'):
        # filters依次指的是：陷波，带通，高通，平滑滤波的点数，如果该选项为None,则没有这个滤波器
        self.srate = srate
        self.chs = chs
        self.fltparam = fltparam
        self.padL = int(self.srate)
        self.cache = np.zeros((self.chs, self.padL), dtype=eegtype)

        self.rawdata = None
        self.ndata = None
        self.bdata = None
        self.hdata = None
        self._genFilters()

    def _genFilters(self):
        '''
        构造指定频率的滤波器,陷波，0.5hz高通，0.5-45带通
        '''
        fs = self.srate/2.
        # 陷波
        self.nflt = signal.butter(N=2, Wn=[self.fltparam[0][0]/ fs, self.fltparam[0][1]/ fs], btype='stop')
        self.hflt = signal.butter(N=2, Wn=self.fltparam[2][0] / fs, btype='highpass')
        self.bflt = signal.butter(N=2, Wn=[self.fltparam[1][0] / fs, self.fltparam[1][1] / fs], btype='bandpass')

    def update(self, fdat): # fdat为一个多行序列

        r,c = fdat.shape
        self.ndata = signal.filtfilt(self.nflt[0],self.nflt[1], fdat)         # notch filter
        self.hdata = signal.filtfilt(self.hflt[0],self.hflt[1],self.ndata)    # high pass
        self.bdata = signal.filtfilt(self.bflt[0], self.bflt[1], self.ndata)  # band pass

        return self.bdata


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
        self.totalChs =  self.emgChs + self.accChs + self.gloveChs + 2

        dataBuffer = buffer[4*headlen:]
        L = int((len(dataBuffer)//(self.totalChs*self.adctype.itemsize))*(self.totalChs*self.adctype.itemsize))
        sampleN = L//(self.totalChs*self.adctype.itemsize)
        dataBuffer = dataBuffer[:L]
        adcData = np.frombuffer(dataBuffer,dtype=self.adctype)
        data = adcData.reshape(sampleN,self.totalChs).transpose()
        return {'srate':self.srate,'emgchs':self.emgChs,'accchs':self.accChs,'glovechs':self.gloveChs,'data':data}


def read_txt_to_matrix(file_path):
    """
    从文本文件中读取数据，每行8个数据，共有N行，并将其转换为8xN的矩阵
    :param file_path: 文本文件路径
    :return: 返回形状为(8, N)的NumPy数组
    """
    # 从文件读取数据
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # 将每行数据转换为列表
    data = []
    for line in lines:
        # 去除换行符并按空格分割
        row = list(map(float, line.strip().split()))
        if len(row) != 8:
            raise ValueError("每行必须包含8个数据")
        data.append(row)
    
    # 转换为NumPy数组并转置为(8, N)
    matrix = np.array(data).T
    return matrix


class EMGSignalPlotter:
    def __init__(self, C=8, F=500):
        # 共有参数设置
        # plt.rcParams = ["Times New Romans"]
        self.C, self.T = C, F

    def plot_time_domain(self, signal):
        """
        绘制时域波形，每个通道单独一个子图，共8个子图（2x4排列）
        """
        if self.C != 8:
            raise ValueError("输入信号必须包含 8 通道以绘制 8 个子图")

        plt.figure(figsize=(12, 8))
        for i in range(self.C):
            plt.subplot(2, 4, i + 1)
            plt.plot(signal[i], color='b')
            plt.title(f'Channel {i + 1}')
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_one_channel_signal(self, signals):
        plt.figure()
        plt.plot(signals)
        plt.show()

    def plot_feature_map(self, features):
        """
        绘制特征图，特征图形状为 (C, T)，右侧添加颜色条，采用 jet 配色
        :param features: 输入特征图，形状为 (C, T)
        """
        plt.figure(figsize=(10, 6))
        plt.imshow(features, aspect='auto', cmap='jet')
        plt.colorbar(label='Feature Value')
        plt.title('Feature Map')
        plt.xlabel('Time')
        plt.ylabel('Channel')
        plt.show()


    def plot_feature_maps_with_colorbar(self, data):
        """
        绘制 F 个特征图，排列为 6x6 网格。
        每个子图中有 C 条折线，每条折线对应 C 维度的一行数据。
        y 轴范围根据数据的实际范围动态调整。

        参数:
            data (np.ndarray): 输入数据，形状为 (F, C, N)。
        """
        F, C, N = data.shape
        assert F == 36, "F 必须等于 36"
        
        # 创建 6x6 的子图
        fig, axes = plt.subplots(6, 6, figsize=(20, 20))
        fig.suptitle("Feature Maps (6x6) - C Lines per Subplot (Dynamic Y-axis)", fontsize=16)
        
        # 定义颜色和样式
        colors = plt.cm.viridis(np.linspace(0, 1, C))  # 使用 viridis 颜色映射
        linestyles = ['-', '--', '-.', ':']  # 定义不同线型
        
        # 绘制每个特征图
        for f in range(F):
            row, col = f // 6, f % 6
            ax = axes[row, col]
            feature_map = data[f]  # 获取当前特征图 (C, N)
            
            # 计算当前子图的 y 轴范围
            y_min = np.min(feature_map)
            y_max = np.max(feature_map)
            
            # # 如果 y 轴范围太小，稍微扩展一点以避免折线被压缩
            # if y_max - y_min < 1e-6:
            #     y_min -= 0.1
            #     y_max += 0.1
            
            # 设置 y 轴范围
            ax.set_ylim(y_min, y_max)
            
            # 绘制每行数据（C 维度）
            for c in range(C):
                # 设置颜色和线型
                color = colors[c]
                linestyle = linestyles[c % len(linestyles)]
                # 绘制当前行数据
                ax.plot(feature_map[c], color=color, linestyle=linestyle, label=f'C={c + 1}')
            
            ax.set_title(f'Feature Map {f + 1}')
            ax.set_xlabel('N')
            ax.set_ylabel('Value')
            ax.legend(loc='upper right', fontsize=6)  # 添加图例
        
        # 调整布局
        plt.tight_layout()
        plt.show()


    def start(self):
        pass


class EMGSignalDecode:
    def __init__(self, fs=500):
        # 所有的输入均是二维信号(C通道, T采样点数)
        self.fs = fs

    def _sampen(self, signal, m=2, r=0.2):
        N = len(signal)
        B = 0
        A = 0
        for i in range(N - m):
            for j in range(i+1, N - m):
                if np.max(np.abs(signal[i:i+m] - signal[j:j+m])) <= r:
                    B += 1
                if np.max(np.abs(signal[i:i+m+1] - signal[j:j+m+1])) <= r:
                    A += 1
        return -np.log((A + 1e-10) / (B + 1e-10)) if B != 0 else 0

    def _ar_coefficients(self, signal, order=4):
        n = len(signal)
        H = hankel(signal[:n-order], signal[n-order-1:])
        y = signal[order:]
        coeffs = np.linalg.lstsq(H, y, rcond=None)[0]
        return coeffs
    
    def _frequency_domain_features(self, signal, fs=500):
        # Compute PSD (Power Spectral Density)
        freqs, psd = welch(signal, fs=fs, nperseg=min(256, len(signal)))

        # Mean Frequency
        mean_freq = np.sum(freqs * psd) / np.sum(psd)

        # Median Frequency
        cumsum_psd = np.cumsum(psd)
        median_freq_index = np.where(cumsum_psd >= 0.5 * cumsum_psd[-1])[0][0]
        median_freq = freqs[median_freq_index]

        # Peak Frequency
        peak_freq_index = np.argmax(psd)
        peak_freq = freqs[peak_freq_index]

        # Total Power
        total_power = np.sum(psd)

        # Spectral Moments
        spectral_moment_1 = np.sum(freqs * psd) / total_power
        spectral_moment_2 = np.sum(freqs**2 * psd) / total_power
        spectral_moment_3 = np.sum(freqs**3 * psd) / total_power

        # Frequency Ratio
        freq_ratio = np.sum(psd[freqs < fs/4]) / np.sum(psd[freqs >= fs/4])

        # Power Spectrum Ratio
        power_ratio = np.sum(psd[freqs < fs/2]) / np.sum(psd[freqs >= fs/2])

        # Variance of Central Frequency
        var_central_freq = np.sum((freqs - spectral_moment_1)**2 * psd) / total_power

        return [mean_freq, median_freq, peak_freq, total_power, spectral_moment_1, spectral_moment_2,
                spectral_moment_3, freq_ratio, power_ratio, var_central_freq]

    def calculate_EMG_features(self, signal):
        # 提取EMG信号特征
        # 输入形状: (C, T), 其中 C 是通道数，T 是采样点数
        # 返回: 每个通道的特征向量
        C, T = signal.shape
        features = []

        for channel in range(C):
            x = signal[channel, :]
            feature_vector = []

            # 0. Mean Absolute Value (MAV)
            mav = np.mean(np.abs(x))
            feature_vector.append(mav)

            # 1. Variance of EMG
            variance = np.var(x)
            feature_vector.append(variance)

            # 2. Root Mean Square (RMS)
            rms = np.sqrt(np.mean(x**2))
            feature_vector.append(rms)

            # 3. Waveform Length (WL)
            wl = np.sum(np.abs(np.diff(x)))
            feature_vector.append(wl)

            # 4. Difference Absolute Mean Value (DAMV)
            damv = np.mean(np.abs(np.diff(x)))
            feature_vector.append(damv)

            # 5. Difference Absolute Standard Deviation Value (DASDV)
            dasdv = np.sqrt(np.sum(np.diff(x)**2) / (T-1))
            feature_vector.append(dasdv)

            # 6. Zero Crossing (ZC)
            zc = np.sum(np.diff(np.sign(x)) != 0) / T
            feature_vector.append(zc)

            # 7. Myopulse Percentage Rate (MYOP)
            threshold = 0.1 * np.max(np.abs(x))
            myop = np.sum(np.abs(x) > threshold) / T
            feature_vector.append(myop)

            # 8. Willison Amplitude (WAMP)
            wamp_threshold = 0.01
            wamp = np.sum(np.abs(np.diff(x)) > wamp_threshold) / (T-1)
            feature_vector.append(wamp)

            # 9. Slope Sign Change (SSC)
            ssc_threshold = 0.01
            ssc = 0
            for i in range(1, T-1):
                delta1 = x[i] - x[i-1]
                delta2 = x[i] - x[i+1]
                if (delta1 * delta2 < 0) and (np.abs(delta1) > ssc_threshold or np.abs(delta2) > ssc_threshold):
                    ssc += 1
            ssc /= T
            feature_vector.append(ssc)

            # 10. Sample Entropy (SampEn)
            sampen_value = self._sampen(x)
            feature_vector.append(sampen_value)

            # 11. Histogram of EMG (10 bins)
            hist, _ = np.histogram(x, bins=10)
            hist = hist / np.sum(hist)  # Normalize
            feature_vector.extend(hist)

            # 12. Auto-Regressive Coefficients (4th Order)
            ar_coeffs = self._ar_coefficients(x)
            feature_vector.extend(ar_coeffs)

            # 提取频域特征
            freq_features = self._frequency_domain_features(x, fs=self.fs)
            feature_vector.extend(freq_features)
            features.append(feature_vector)

        return np.array(features)


class PreProcessManager:
    # 采样率为500Hz
    def __init__(self, sample_rate=500):
        self.sample_rate = sample_rate
        self.filter = ButterFilter()

    def data_filter(self, x):
        self.filter.reset(srate=500, chs=8,
                           fltparam=[(49, 51), (20, 150), (1, 0), None],eegtype=EEGTYPE)
        xs = []
        for i in range(x.shape[0]):
            rdat = self.filter.update(x[i, :8, :].T).T
            xs.append(np.concatenate((x[i, :8, :], x[i, 8:, :])))
        xs = np.array(xs)

        print(xs.shape)
        return xs
    
    def data_normalize(self, x):
        # z-method 数据归一化
        return x

    
    def data_slice(self, x, window_size, window_mov_t):
        # 根据窗口大学和长度执行数据分段
        # 只检查，初始点和结束点的标签，标签一致即为样本
        s_x, s_y = [], []
        idx_start = 0
        while idx_start < x.shape[1]:
            idx_end = idx_start + int(self.sample_rate * window_size)
            if idx_end >= x.shape[1]: break
            if x[-1][idx_start] == x[-1][idx_end]:
                s_x.append(x[:-2, idx_start: idx_end])
                s_y.append(x[-1][idx_start])
            idx_start = idx_start + int(self.sample_rate * window_mov_t)
        s_x, s_y = np.array(s_x), np.array(s_y)
        return s_x, s_y
    
    def data_preprocess_all(self, x, window_size, window_mov_t):
        x, y = self.data_slice(x, window_size, window_mov_t)
        x = self.data_filter(x)
        x_train, y_train, x_test, y_test = self.divide_train_test_data(x, y, classes=3)
        return x_train, y_train, x_test, y_test

    def divide_train_test_data(self, x, y, classes):
        num = x.shape[0]
        critical_value_tmp = int(num * 2 / 3)
        bias = 50
        for idx in range(critical_value_tmp - bias, critical_value_tmp + bias):
            if y[idx] == (classes-1) and y[idx+1] == 0: 
                critical_value = idx
        
        x_train, y_train = x[: critical_value, ...], y[: critical_value, ...]
        x_test, y_test = x[critical_value: , ...], y[critical_value: , ...]

        return x_train, y_train, x_test, y_test





def normalize_zscore(data):
    """
    按 N 维度进行 Z-score 归一化。

    参数:
        data (np.ndarray): 输入数据，形状为 (F, C, N)。

    返回:
        np.ndarray: 归一化后的数据。
    """
    # 计算每个 N 维度的均值和标准差
    mean = np.mean(data, axis=(0, 1, 2), keepdims=True)
    std = np.std(data, axis=(0, 1, 2), keepdims=True)
    # 标准化
    normalized_data = (data - mean) / std
    return normalized_data







def detect_and_replace_spikes(data, threshold):
    """
    检测并替换突刺异常值。

    参数:
        data (np.ndarray): 输入数据，形状为 (C, T)。
        threshold (float): 突刺检测的阈值。

    返回:
        np.ndarray: 处理后的数据。
    """
    C, T = data.shape
    processed_data = data.copy()

    for c in range(C):
        for t in range(1, T - 1):
            # 计算前后差值
            diff_prev = data[c, t] - data[c, t - 1]
            diff_next = data[c, t] - data[c, t + 1]
            
            # 如果前后差值均超过阈值，则认为是突刺
            if diff_prev > threshold and diff_next > threshold:
                # 用两侧的平均值替换
                processed_data[c, t] = (data[c, t - 1] + data[c, t + 1]) / 2
    
    return processed_data

if __name__ == "__main__":
    # 示例信号
    # file_path = os.path.join(os.getcwd(), "Data", "PublicData", "example.txt") # 替换为您的文件路径
    # matrix = read_txt_to_matrix(file_path)

    file_reader = ReadGmData(os.path.join(os.getcwd(), "Data", "xuxuechao", "0307", "trial_01.dat"))

    data = file_reader.readfile()["data"]

    preprocess_manager = PreProcessManager()


    x_train, y_train, x_test, y_test = preprocess_manager.data_preprocess_all(data, 0.5, 0.25)

    # filter = ButterFilter()
    # filter.reset(srate=500, chs=8,
    #                        fltparam=[(49, 51), (20, 150), (1, 0), None],eegtype=EEGTYPE)


    

    plotter = EMGSignalPlotter()

    # plotter.plot_one_channel_signal(data[4, :])




    # plotter.plot_feature_map()
    # plotter.plot_time_domain(data[:8, :])

    # sample_features = []
    # for i in range(50):
    #     print(f"第{i}个样本处理！")
    #     features = data[:8, i*250: i*250+250]
    #     features = filter.update(features)



    #     signal_decoder = EMGSignalDecode()

    #     features = signal_decoder.calculate_EMG_features(features)


    #     sample_features.append(features)

    # sample_features = np.array(sample_features).transpose(2, 1, 0)

    # # sample_features = normalize_zscore(sample_features)

    # plotter = EMGSignalPlotter()

    # plotter.plot_feature_maps_with_colorbar(sample_features[:, :, 2:])

    # plotter.plot_time_domain(features[:, 5:20])
    # plotter.plot_feature_map(features[:, 5:20])




