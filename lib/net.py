from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))

class Net(nn.Module):
    def __init__(self, width=(48, 96, 96, 128), n_extra=2, lag=0):
        super().__init__()
        w1, w2, w3, w4 = width
        self.lag = lag
        self.conv = nn.Sequential(
            nn.Conv1d(4, w1, 7, stride=2, padding=3, bias=False), nn.BatchNorm1d(w1), nn.ReLU(inplace=True),
            nn.Conv1d(w1, w2, 5, stride=2, padding=2, bias=False), nn.BatchNorm1d(w2), nn.ReLU(inplace=True),
            nn.Conv1d(w2, w3, 3, stride=2, padding=1, bias=False), nn.BatchNorm1d(w3), nn.ReLU(inplace=True),
            nn.Conv1d(w3, w4, 3, stride=1, padding=1, bias=False), nn.BatchNorm1d(w4), nn.ReLU(inplace=True),
        )
        n_lag = 0 if not lag else len(PAIRS) * (2 * lag + 1)
        n_stats = 4 + 4 + n_extra + n_lag
        if lag:
            self.stats = nn.Sequential(nn.Linear(n_stats, 64), nn.ReLU(inplace=True),
                                       nn.Linear(64, 24), nn.ReLU(inplace=True))
        else:
            self.stats = nn.Sequential(nn.Linear(n_stats, 24), nn.ReLU(inplace=True))
        self.head = nn.Sequential(
            nn.Linear(2 * w4 + 24, 128), nn.ReLU(inplace=True), nn.Dropout(0.1), nn.Linear(128, 8),
        )

    def _xlag(self, xn: torch.Tensor) -> torch.Tensor:
        k = self.lag
        B, _, L = xn.shape
        ia = [i for i, _ in PAIRS]
        ib = [j for _, j in PAIRS]
        win = F.pad(xn[:, ib, :], (k, k)).unfold(-1, L, 1)
        c = (xn[:, ia, :].unsqueeze(2) * win).sum(dim=-1)
        den = (xn * xn).sum(dim=-1)
        d = torch.sqrt(den[:, ia] * den[:, ib]) + 1e-9
        return (c / d.unsqueeze(2)).reshape(B, -1)

    def forward(self, x: torch.Tensor, extra: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=-1, keepdim=True)
        s = x.std(dim=-1, keepdim=True) + 1e-6
        g = s.mean(dim=1, keepdim=True) + 1e-6
        xn = (x - mu) / g
        h = self.conv(xn)
        pooled = torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=1)
        acf1 = (xn[:, :, :-1] * xn[:, :, 1:]).mean(dim=-1)
        ratio = torch.log(s[:, :, 0] / g[:, 0, :])
        parts = [ratio, acf1, extra]
        if self.lag:
            parts.append(self._xlag(xn))
        st = self.stats(torch.cat(parts, dim=1))
        return self.head(torch.cat([pooled, st], dim=1))
