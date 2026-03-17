Code and data for paper Real-Time Gesture Recognition Using Multi-Channel sEMG Signals with Spatial-Temporal Spiking Neural Network
============================================================

本仓库实现了基于 sEMG 信号的6类手势/动作分类，包括：

- **STSNN**：在 L-EMGNet 拓扑基础上引入脉冲神经元（SNN），探索类脑计算在 sEMG 分类中的性能。

数据为多导联 sEMG + 标签，按窗口滑动分段，并支持单日 / 跨日两类实验设置。


目录结构
--------

- `get_data.py`：读取原始 `raw.dat` 生物信号文件并进行预处理与切片。
  - 读取自定义二进制格式，解析采样率与通道数等头信息。
  - 对前 **8 导联 EMG 通道**做 **20–150 Hz 六阶巴特沃斯带通滤波**（零相位 `filtfilt`）。
  - 采用 **500 ms 窗口 / 250 ms 滑动步长** 对数据按标签切片：
    - 采样率 500 Hz → 窗口长度 250 点、步长 125 点。
    - 仅保留窗口内标签全相同的片段。
  - 输出：
    - `segments.npy`：形状 `(N, 8, 250)` 的数据片段。
    - `segments_labels.npy`：长度为 `N` 的标签数组。

- `model.py`：模型定义。
  - `STSNN`：
    - 与 L-EMGNet **拓扑完全对应**，仅将 `ELU` 激活替换为 **ParametricLIFNode**（SpikingJelly）。
    - 使用 `step_mode='m'`，将时间维视为步数，对 2D 特征 `(N, C, H, W)` 在时间维展开为 `(T, N, C, H)` 送入 LIF，再还原。
    - 目标是在保留 EMGNet 表达能力的基础上，引入脉冲神经元，尽量让性能接近 EMGNet。

- `main.py`：训练与评估入口。
  - 支持 **两种模型**：
    - `--model stsnn`：使用 STSNN（EMGNet + SNN）。
  - 支持 **两类实验设置**：
    - **单日实验（within-day）**：
      - 对每个被试的 Day1 和 Day2：
        - 将对应日的全部片段与标签 **先打乱**（使用带种子的随机置乱，保证可复现）。
        - 再按 **80% 训练 / 20% 测试** 划分。
      - 对每个被试，分别计算 Day1 与 Day2 的测试准确率，取二者平均作为该被试单日结果。
      - 最终报告所有被试的单日平均准确率。
    - **跨日实验（cross-day）**：
      - 对每个被试：
        - **Day1 全部数据训练**，**Day2 全部数据测试**。
        - 标签空间以 Day1 的类别为基准，对 Day2 原始标签做对齐映射。
      - 报告每个被试的跨日准确率以及所有被试的平均跨日准确率。
  - 训练细节：
    - 使用 `tqdm` 展示 epoch 与 batch 级进度条。
    - 损失函数：`CrossEntropyLoss`。
    - 优化器：`Adam`，学习率可由 `--lr` 设置。
    - 在 STSNN 上训练时，每个 batch、每次评估前调用 `reset_net()` 清除网络状态。


环境依赖
--------

建议使用 Python 3.8+ 环境。主要依赖：

- `numpy`
- `scipy`（用于巴特沃斯带通滤波）
- `torch`（PyTorch）
- `tqdm`
- `spikingjelly`（仅 STSNN 需要）

示例安装（使用 pip）：

```bash
pip install numpy scipy torch tqdm spikingjelly
```


数据准备
--------

默认数据目录结构为：

```text
data/
  Sub01/
    Day1/
      raw.dat
    Day2/
      raw.dat
  Sub02/
    Day1/
      raw.dat
    Day2/
      raw.dat
  ...
  Sub10/
    Day1/
    Day2/
```

每个 `raw.dat` 为自定义格式的生物信号二进制文件，头部包含采样率和通道数等信息。

运行 `get_data.py` 后，将在对应 `DayX` 目录下生成：

- `segments.npy`：`(N, 8, 250)`。
- `segments_labels.npy`：`(N,)`。


运行实验
--------

1. **生成分段数据**

在项目根目录执行：

```bash
python get_data.py
```

将遍历 `Sub01`–`Sub10` 与 `Day1`、`Day2`，对每个 `raw.dat` 进行滤波与切片并保存。

2. **单日 + 跨日实验（默认使用 EMGNet）**

```bash
python main.py
```

这会：

- 对所有被试运行单日实验（Day1/Day2，80/20 划分）。
- 对所有被试运行跨日实验（Day1 训练，Day2 测试）。
- 在控制台输出：
  - 每个被试的单日结果（Day1 准确率、Day2 准确率、平均）。
  - 每个被试的跨日准确率。
  - 两种实验的整体平均准确率。

3. **仅运行单日实验**

```bash
python main.py --single_day
```

4. **仅运行跨日实验**

```bash
python main.py --cross_day
```

5. **模型训练与超参数示例**

```bash

# 使用 STSNN（EMGNet 拓扑 + LIF），并适当减小 Dropout
python main.py --model stsnn --dropout 0.3 --epochs 100 --lr 1e-3
```

许可与引用
----------

- 本代码可用于科研与教学用途，若在论文或项目中使用，请在致谢中提及本仓库。

