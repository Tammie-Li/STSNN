import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


def bandpass_filter(signals, sample_rate, low_hz=20.0, high_hz=150.0, order=6):
    """
    对多通道信号做 20–150 Hz 巴特沃斯六阶带通滤波（零相位 filtfilt）。
    signals: (n_frames, n_channels), 沿第 0 维为时间。
    """
    nyq = 0.5 * sample_rate
    low = max(low_hz / nyq, 1e-9)
    high = min(high_hz / nyq, 1.0 - 1e-9)
    b, a = butter(order, [low, high], btype="bandpass")
    out = np.empty_like(signals, dtype=np.float64)
    for ch in range(signals.shape[1]):
        out[:, ch] = filtfilt(b, a, signals[:, ch].astype(np.float64), axis=0)
    return out.astype(signals.dtype)


def read_data_file(filename):
    """读取自定义格式的生物信号数据文件，返回二维数组"""
    with open(filename, 'rb') as f:
        header_data = np.fromfile(f, dtype=np.int32, count=7)
        header = {
            'header_len': header_data[0], 'file_version': header_data[1],
            'data_type': 'float32' if header_data[2] == 2 else 'float64',
            'sample_rate': header_data[3], 'emg_chs': header_data[4],
            'acc_chs': header_data[5], 'glove_chs': header_data[6]
        }
        
        frame_size = header['emg_chs'] + header['acc_chs'] + header['glove_chs'] + 2
        dtype = np.float32 if header['data_type'] == 'float32' else np.float64
        
        raw_frames = np.fromfile(f, dtype=dtype)
        n_frames = len(raw_frames) // frame_size
        data = raw_frames[:n_frames * frame_size].reshape(n_frames, frame_size)
    return header, data


def segment_by_label(data, window_ms=500.0, step_ms=250.0, sample_rate=500, n_channels=8):
    window_len = int(round(window_ms * sample_rate / 1000.0))   
    step_len = int(round(step_ms * sample_rate / 1000.0))       
    signals = data[:, :n_channels].astype(np.float32)  
    lbl = data[:, -1]                            
    n_frames, C = signals.shape
    if n_frames < window_len:
        return np.empty((0, C, window_len), dtype=np.float32), np.array([], dtype=lbl.dtype)
    segments_list = []
    labels_list = []
    for start in range(0, n_frames - window_len + 1, step_len):
        end = start + window_len
        win_labels = lbl[start:end]
        if np.all(win_labels == win_labels[0]):
            segments_list.append(signals[start:end].T)   # (C, T)
            labels_list.append(win_labels[0])
    if not segments_list:
        return np.empty((0, C, window_len), dtype=np.float32), np.array([], dtype=lbl.dtype)
    segments = np.stack(segments_list, axis=0)   # (N, C, T)
    labels = np.array(labels_list, dtype=lbl.dtype)
    return segments, labels


def segment_and_save(data, save_path, window_ms=500.0, step_ms=250.0, sample_rate=500,
                     n_channels=8, save_labels=True, filter_hz=(20.0, 150.0), filter_order=6):
    """
    filter_hz: (low, high) 带通范围，单位 Hz；None 表示不过滤。
    filter_order: 巴特沃斯阶数，默认 6。
    """
    data = np.asarray(data, dtype=np.float64)
    if filter_hz is not None:
        low_hz, high_hz = filter_hz
        data[:, :n_channels] = bandpass_filter(
            data[:, :n_channels], sample_rate,
            low_hz=low_hz, high_hz=high_hz, order=filter_order
        )
    segments, labels = segment_by_label(data, window_ms=window_ms, step_ms=step_ms,
                                        sample_rate=sample_rate, n_channels=n_channels)
    os.makedirs(os.path.dirname(os.path.abspath(save_path)) or '.', exist_ok=True)
    np.save(save_path, segments)
    if save_labels:
        base, ext = os.path.splitext(save_path)
        label_path = base + '_labels' + (ext or '.npy')
        np.save(label_path, labels)
    return segments, labels


# ====================== 函数调用示例 ======================
if __name__ == "__main__":
    for sub in range(1, 11):
        for day in range(1, 3):  # Day1, Day2，供 main.py 单日/跨日实验使用
            file_path = os.path.join(os.getcwd(), 'data', f'Sub{sub:>02d}', f'Day{day}', "raw.dat")
            header, data = read_data_file(file_path)
            save_path = os.path.join(os.getcwd(), 'data', f'Sub{sub:>02d}', f'Day{day}', 'segments.npy')
            segments, labels = segment_and_save(
                data,
                save_path,
                window_ms=500.0,
                step_ms=250.0,
                sample_rate=500,
                save_labels=True,
            )
            print(f"片段形状 (N,C,T) = {segments.shape}, 标签形状 (N,) = {labels.shape}")