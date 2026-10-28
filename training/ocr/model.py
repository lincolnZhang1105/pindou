"""CRNN + CTC：图例色号、颗数、坐标数字共用一个小模型（< 1 MB）。"""
from __future__ import annotations

import torch
from torch import nn

from training.ocr.common import INPUT_H, NUM_CLASSES


def _block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class CRNN(nn.Module):
    """输入 [N, 1, 32, W]，输出 [T = W/4, N, C] 的 log 概率（C 含 blank）。"""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.cnn = nn.Sequential(
            _block(1, 32), nn.MaxPool2d(2, 2),          # 16 × W/2
            _block(32, 64), nn.MaxPool2d(2, 2),         # 8 × W/4
            _block(64, 96), _block(96, 96), nn.MaxPool2d((2, 1), (2, 1)),   # 4 × W/4
            _block(96, 128), nn.MaxPool2d((2, 1), (2, 1)),                   # 2 × W/4
            nn.Conv2d(128, 128, (2, 1), bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),  # 1 × W/4
        )
        self.rnn = nn.LSTM(128, hidden, bidirectional=True, batch_first=False)
        self.fc = nn.Linear(2 * hidden, NUM_CLASSES)
        assert INPUT_H == 32

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.cnn(x)                    # [N, 128, 1, T]
        f = f.squeeze(2).permute(2, 0, 1)  # [T, N, 128]
        f, _ = self.rnn(f)
        return self.fc(f).log_softmax(-1)  # [T, N, C]
