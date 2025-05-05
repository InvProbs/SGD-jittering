import math, torch, os
import torch.nn as nn
import matplotlib.pyplot as plt
# import pytorch_ssim as pytorch_ssim
from torchmetrics.image import StructuralSimilarityIndexMeasure
import numpy as np
from operators import training_mode as mode

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def add_detail_to_x(X, args):
    if args.dataset == 'MRI':
        Xg = torch.clone(X)
        row, col = [150, 150]
        length = 20
        patch = Xg[:, :, row:row + 1, col:col + length]
        g = torch.ones_like(patch) * 0.7
        Xg[:, :, row:row + 1, col:col + length] = torch.max(g, patch)
        return Xg
    if args.dataset == 'seis':
        Xg = torch.clone(X)
        # row, col = [175, 150] # SGD
        row, col = [140, 170] # SPGD [125, 150]
        length = 10
        patch = Xg[:, :, row:row + 1, col:col + length]
        g = - torch.ones_like(patch) * 0.1  # magnitude of 0.1 and 0.3
        Xg[:, :, row:row + 1, col:col + length] = g  # torch.max(g, patch)
        return Xg
    else:
        Xg = torch.clone(X)
        # row, col = np.random.randint(0, 24, [2])
        row, col = [5, 5]
        patch = Xg[:, :, row:row + 1, col:col + 3]
        g = torch.ones_like(patch) * 0.7
        # g *= args.epsilon / mode.norms_3D(g)
        Xg[:, :, row:row + 1, col:col + 3] = torch.max(g, patch)
        # torch.max(patch + torch.ones_like(patch), torch.ones_like(patch))
        return Xg


def PSNR1chan(Xk, X):  # ONLY the REAL Part
    bs, C, W, H = X.shape
    Xk = Xk[:, 0, :, :]
    X = X[:, 0, :, :]
    mse = torch.sum(((Xk - X) ** 2).reshape(bs, -1), dim=1) / (W * H)
    # return 20 * torch.log10(torch.max(torch.max(X, dim=1)[0], dim=1)[0] / torch.sqrt(mse))
    return -10 * torch.log10(mse)


def compute_metrics1chan(Xk, X, X0):
    Xk, X0 = torch.clamp(Xk, 0, 1), torch.clamp(X0, 0, 1)
    init_psnr, recon_psnr = PSNR1chan(X0, X), PSNR1chan(Xk, X)

    bs = X.shape[0]
    avg_init_psnr = torch.sum(init_psnr) / bs
    avg_recon_psnr = torch.sum(recon_psnr) / bs
    avg_delta_psnr = torch.sum(recon_psnr - init_psnr) / bs

    # Xk = torch.clamp(torch.abs(torch.view_as_complex(Xk.permute(0, 2, 3, 1).contiguous())), min=0, max=1)
    # X = X[:, 0:1, :, :]
    # avg_ssim = pytorch_ssim.SSIM(Xk, X)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    avg_ssim = ssim(Xk.cpu(), X.cpu())
    return avg_init_psnr, avg_recon_psnr, avg_delta_psnr, avg_ssim


def compute_metrics2chan(Xk, X, X0):
    Xk, X0 = torch.clamp(Xk[:, 0:1], 0, 1), torch.clamp(X0[:, 0:1], 0, 1)
    X = torch.clamp(X[:, 0:1], 0, 1)
    init_psnr, recon_psnr = PSNR1chan(X0, X), PSNR1chan(Xk, X)

    bs = X.shape[0]
    avg_init_psnr = torch.sum(init_psnr) / bs
    avg_recon_psnr = torch.sum(recon_psnr) / bs
    avg_delta_psnr = torch.sum(recon_psnr - init_psnr) / bs

    ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    avg_ssim = ssim(Xk.cpu(), X.cpu())
    return avg_init_psnr, avg_recon_psnr, avg_delta_psnr, avg_ssim


def PSNR3chan(Xk, X):
    bs, C, W, H = X.shape
    mse = torch.sum(((Xk - X) ** 2).reshape(bs, -1), dim=1) / (C * W * H)
    return 20 * torch.log10(1 / torch.sqrt(mse))


def compute_metrics3chan(Xk, X, X0):
    init_psnr, recon_psnr = PSNR3chan(X0, X), PSNR3chan(Xk, X)
    bs = X.shape[0]
    avg_init_psnr = torch.sum(init_psnr) / bs
    avg_recon_psnr = torch.sum(recon_psnr) / bs
    avg_delta_psnr = torch.sum(recon_psnr - init_psnr) / bs
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    avg_ssim = ssim(Xk.cpu(), X.cpu())
    return avg_init_psnr, avg_recon_psnr, avg_delta_psnr, avg_ssim


def plot_MRI(Xk, X0, X, criteria, save_path, epoch):
    plt.figure(figsize=(10, 15))
    bs, c, w, h = Xk.shape
    if c != 1:
        x_hat = torch.clamp(torch.abs(torch.view_as_complex(Xk.permute(0, 2, 3, 1).contiguous())), min=0, max=1)
        init = torch.view_as_complex(X0.permute(0, 2, 3, 1).contiguous())
        init_clamp = torch.clamp(torch.abs(init), min=0, max=1)
    else:
        x_hat = Xk[:, 0, :, :]
        init = X0[:, 0, :, :]
        init_clamp = init
    X_true = X[:, 0, :, :]
    err = torch.abs(X_true - x_hat)
    for i in range(min(3, len(X))):
        # ii = i * 5 + 2
        ii = i  # + 3
        plt.subplot(4, 3, i + 1)
        psnr = 20 * math.log10(torch.max(X_true) / math.sqrt(criteria(init_clamp[ii], X_true[ii])))
        plt.imshow(init_clamp[ii].detach().cpu(), cmap='gray')
        plt.title('$x_0$, PSNR = {0:.3f}'.format(psnr))
        plt.axis('off')
        plt.subplot(4, 3, i + 4)
        psnr = 20 * math.log10(torch.max(X_true) / math.sqrt(criteria(x_hat[ii], X_true[ii])))
        plt.imshow(x_hat[ii].detach().cpu(), cmap='gray')
        plt.title('$\hat{x}$, ' + 'PSNR = {0:.3f}'.format(psnr))
        plt.axis('off')
        plt.subplot(4, 3, i + 7)
        plt.imshow(X_true[ii].detach().cpu(), cmap='gray')
        plt.title('Clean image')
        plt.axis('off')
        plt.subplot(4, 3, i + 10)
        plt.imshow(err[ii].detach().cpu(), cmap='gray')
        plt.title('Error, max:{0:.2f}'.format(torch.max(err)))
        plt.axis('off')

    # plt.show()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{epoch}_results.png'))
    plt.close()



def plot_2D(X, Xk, y, save_path, epoch, title=''):
    Xk = Xk.detach().cpu()
    X = X.detach().cpu()
    y = y.detach().cpu()

    # Plot 2D image
    i = 0
    plt.figure(figsize=(10, 5))
    plt.suptitle(title)
    plt.subplot(1, 3, 1)
    plt.imshow(y[i, 0], cmap='gray')
    plt.title('Trace')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(Xk[i, 0], cmap='gray')
    plt.title('Recovered $\hat{X}$')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(X[i, 0], cmap='gray')
    plt.title('Ground Truth')
    plt.axis('off')
    plt.savefig(os.path.join(save_path, f'result_{epoch}_result.png'))
    plt.close()