"""Deep baselines. All take input of shape (B, C=30, T=384) and return logits (B, 2)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class InterpretableCNN(nn.Module):
    """Cui et al., TNNLS 2022.

    bn_mode="batch"   -> per-batch BN as published (track_running_stats=False). Only meaningful when a
                         whole held-out subject is scored as one batch (transductive).
    bn_mode="running" -> BN with running statistics; valid for one-epoch-at-a-time streaming.
    """

    def __init__(self, channels=30, n1=16, kernel_length=64, sample_length=384, classes=2, bn_mode="running"):
        super().__init__()
        self.pointwise = nn.Conv2d(1, n1, (channels, 1))
        self.depthwise = nn.Conv2d(n1, n1 * 2, (1, kernel_length), groups=n1)
        self.relu = nn.ReLU()
        if bn_mode == "batch":
            self.bn = nn.BatchNorm2d(n1 * 2, track_running_stats=False, momentum=0)
        else:
            self.bn = nn.BatchNorm2d(n1 * 2)
        self.gap = nn.AvgPool2d((1, sample_length - kernel_length + 1))
        self.fc = nn.Linear(n1 * 2, classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.bn(self.relu(self.depthwise(self.pointwise(x))))
        return self.fc(self.gap(x).flatten(1))


class EEGNet(nn.Module):
    """EEGNet-8,2 (Lawhern et al., 2018), parameters for 128 Hz input."""

    def __init__(self, channels=30, samples=384, classes=2, f1=8, d=2, f2=16, kernel_length=64, dropout=0.25):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, f1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f1 * d, (channels, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1 * d),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(f1 * d, f1 * d, (1, 16), padding=(0, 8), groups=f1 * d, bias=False),
            nn.Conv2d(f1 * d, f2, 1, bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        with torch.no_grad():
            n_feat = self.block2(self.block1(torch.zeros(1, 1, channels, samples))).numel()
        self.fc = nn.Linear(n_feat, classes)

    def forward(self, x):
        x = self.block2(self.block1(x.unsqueeze(1)))
        return self.fc(x.flatten(1))


# ---------------------------------------------------------------------------
# DeltaGateNet (Po et al., IEEE T-ITS 2026). Vendored unchanged in substance from
# https://github.com/yip-tin-po-alex/DeltaGateNet (commit c156b3a), MIT License,
# Copyright (c) 2026 alexyip.
# ---------------------------------------------------------------------------
class BidirectionalDelta(nn.Module):
    def forward(self, x):
        delta = F.pad(x[:, :, 1:] - x[:, :, :-1], (1, 0))
        return torch.cat([F.relu(delta), F.relu(-delta)], dim=1)


class GatedTemporalConv(nn.Module):
    def __init__(self, input_channels, hidden_dims=16, num_layers=2, dropout=0.5, kernel_size=7):
        super().__init__()
        self.hidden_dims = hidden_dims
        width = input_channels * hidden_dims
        self.input_proj = nn.Conv1d(input_channels, width, kernel_size=1)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(width, width, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(width),
                nn.GELU(),
                nn.Conv1d(width, width, kernel_size=1),
                nn.Dropout(dropout),
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dims)

    def forward(self, x):
        b, c, t = x.shape
        x = self.input_proj(x)
        for layer in self.layers:
            x = x + layer(x)
        return self.norm(x.view(b, c, self.hidden_dims, t).mean(dim=-1))


class DeltaGateNet(nn.Module):
    def __init__(self, num_channels=30, num_classes=2, hidden_dims=16):
        super().__init__()
        self.temporal_diff = BidirectionalDelta()
        self.conv = GatedTemporalConv(2 * num_channels, hidden_dims)
        layers, d_in = [], 2 * num_channels * hidden_dims
        for _ in range(3):
            layers += [nn.Linear(d_in, hidden_dims), nn.BatchNorm1d(hidden_dims), nn.LeakyReLU(0.3), nn.Dropout(0.5)]
            d_in = hidden_dims
        self.mlp = nn.Sequential(*layers, nn.Linear(hidden_dims, num_classes))
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv(self.temporal_diff(x))
        return self.mlp(x.flatten(1))
