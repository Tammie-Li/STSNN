import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, surrogate, functional


class STSNN(nn.Module):
    """
    在 EMGNet 结构基础上加入 SNN；TC 为 (channel×time_point) 即 8×9 联合卷积。
    卷积前在导联维（高）上补零，使 8×9 核滑动后导联维仍为 channel，不塌缩为 1。
    """
    def __init__(self, channel=8, time_length=250, num_classes=6, drop_out=0.4,
                 time_point=9, N_t=8, N_s=16):
        super(STSNN, self).__init__()
        self.channel = channel
        self.time_length = time_length
        self.N_t = N_t
        self.N_s = N_s
        self._pad_ch = channel - 1
        self.block_1_conv = nn.Conv2d(
            1, N_t, (channel, time_point), padding=(0, time_point // 2), bias=False
        )
        self.block_1_bn = nn.BatchNorm2d(N_t)
        self.block_1_lif = neuron.ParametricLIFNode(
            init_tau=2.0, surrogate_function=surrogate.Sigmoid(), step_mode="m"
        )
        # Block 2：深度空间卷积 (channel, 1)，与 EMGNet 一致
        self.block_2_conv = nn.Conv2d(N_t, N_s, (channel, 1), groups=N_t, bias=False)
        self.block_2_bn = nn.BatchNorm2d(N_s)
        self.block_2_lif = neuron.ParametricLIFNode(
            init_tau=2.0, surrogate_function=surrogate.Sigmoid(), step_mode="m"
        )
        self.block_2_pool = nn.AvgPool2d((1, 4))
        self.block_2_drop = nn.Dropout(drop_out)
        # Block 3：深度可分离卷积 + BN + LIF + 池化 + Dropout
        self.block_3_dw = nn.Conv2d(N_s, N_s, (1, N_s), padding=(0, N_s // 2), groups=N_s, bias=False)
        self.block_3_pw = nn.Conv2d(N_s, N_s, (1, 1), bias=False)
        self.block_3_bn = nn.BatchNorm2d(N_s)
        self.block_3_lif = neuron.ParametricLIFNode(
            init_tau=2.0, surrogate_function=surrogate.Sigmoid(), step_mode="m"
        )
        self.block_3_pool = nn.AvgPool2d((1, 8))
        self.block_3_drop = nn.Dropout(drop_out)
        self.fc_dim = N_s * 1 * 8
        self.fc = nn.Linear(self.fc_dim, num_classes)

    def _lif_forward_2d(self, x, lif):
        """将 (N, C, H, W) 按时间维 W 展开为 (W, N, C, H)，过 LIF 再还原。"""
        # x: (N, C, H, W) -> (W, N, C, H)
        x = x.permute(3, 0, 1, 2)
        x = lif(x)
        x = x.permute(1, 2, 3, 0)
        return x

    def forward(self, x):
        if x.size(-1) == 250:
            x = F.pad(x, (0, 6))
        x = x.unsqueeze(1)
        ph = self._pad_ch
        pt, pb = ph // 2, ph - ph // 2
        x = F.pad(x, (0, 0, pt, pb))
        # Block 1 -> (N, N_t, channel, T)
        x = self.block_1_conv(x)
        x = self.block_1_bn(x)
        x = self._lif_forward_2d(x, self.block_1_lif)
        # Block 2
        x = self.block_2_conv(x)
        x = self.block_2_bn(x)
        x = self._lif_forward_2d(x, self.block_2_lif)
        x = self.block_2_pool(x)
        x = self.block_2_drop(x)
        # Block 3
        x = self.block_3_dw(x)
        x = self.block_3_pw(x)
        x = self.block_3_bn(x)
        x = self._lif_forward_2d(x, self.block_3_lif)
        x = self.block_3_pool(x)
        x = self.block_3_drop(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def reset_net(self):
        functional.reset_net(self)
