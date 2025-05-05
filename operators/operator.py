import numpy as np
import torch, numbers, math
import torch.nn as nn
from scipy.fftpack import dct, idct
import random

def normalize(X, bs):
    maxVal, _ = torch.max(X.reshape(bs, -1), dim=1)
    minVal, _ = torch.min(X.reshape(bs, -1), dim=1)
    return (X - minVal[:, None, None]) / (maxVal - minVal)[:, None, None]


def normalize_seis(yn, X, bs):
    if X is None:
        maxVal, _ = torch.max(torch.abs(yn.reshape(bs, -1)), dim=1)
        maxVal[maxVal == 0] = 1
        if len(yn.shape) == 3:
            return maxVal, yn / maxVal[:, None, None, None]
        return maxVal, yn / maxVal[:, None, None, None]
    else:
        maxVal, _ = torch.max(torch.abs(yn.reshape(bs, -1)), dim=1)
        if len(X.shape) == 3:
            return maxVal, yn / maxVal[:, None, None, None], X / maxVal[:, None, None]
        return maxVal, yn / maxVal[:, None, None, None], X / maxVal[:, None, None, None]

class LinearOperator(torch.nn.Module):
    def __init__(self):
        super(LinearOperator, self).__init__()

    def forward(self, x):
        pass

    def adjoint(self, x):
        pass

    def gramian(self, x):
        return self.adjoint(self.forward(x))


class SelfAdjointLinearOperator(LinearOperator):
    def adjoint(self, x):
        return self.forward(x)


class Identity(SelfAdjointLinearOperator):
    def forward(self, x):
        return x


class OperatorPlusNoise(torch.nn.Module):
    def __init__(self, operator, noise_sigma):
        super(OperatorPlusNoise, self).__init__()
        self.internal_operator = operator
        self.noise_sigma = noise_sigma

    def forward(self, x):
        A_x = self.internal_operator(x)
        return A_x + self.noise_sigma * torch.randn_like(A_x)


class convolution(LinearOperator):
    def __init__(self, W, trace_length):
        super(convolution, self).__init__()
        self.A = W  # Wavelet Matrix
        self.trace_len = trace_length

    def forward(self, x):
        return self.A @ x

    def gramian(self, x):
        return self.A.T @ self.A @ x

    def adjoint(self, y):
        return self.A.T @ y