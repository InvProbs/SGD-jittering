import random, torch, torchvision, cv2, glob, json, os, scipy
import numpy as np
from operators.operator import *

""" for seismic deconvolution problem, generate paired data """
def gen_conv_mtx():
    W = scipy.signal.ricker(51, 4)
    W /= np.max(W)
    W = scipy.linalg.convolution_matrix(W.squeeze(), 352)
    pad = (len(W) - 352) // 2
    W = torch.Tensor(W[pad:len(W) - pad, :])
    return W

