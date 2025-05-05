import torch.nn as nn
import torch

class DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=17, features=64):
        super().__init__()
        kernel_size = 3
        padding = 1
        layers = [nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding,
                      bias=False),
                  nn.ReLU(inplace=True)]
        for _ in range(num_of_layers - 2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                          bias=False))
            layers.append(nn.GroupNorm(4, features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding,
                      bias=False))
        self.dncnn = nn.Sequential(*layers)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            module.weight.data.normal_(mean=0.0, std=0.01)
            if module.bias is not None:
                module.bias.data.zero_()
    def forward(self, x):
        out = self.dncnn(x) + x
        return out

class MRI_gd(nn.Module):
    def __init__(self, linear_op, DnCNN, args):
        super(MRI_gd, self).__init__()
        self.eta = nn.Parameter(torch.ones(1) * args.eta)
        self.linear_op = linear_op
        self.R = DnCNN
        self.maxLayers = args.maxiters
        self.sigma = args.sigma_wk if args.train_mode == 'sgd-jitter' and args.train else 0
    def _linear_op(self, x):
        return self.linear_op.forward(x)

    def _linear_adjoint(self, x):
        return self.linear_op.adjoint(x)

    def set_initial_point(self, y):
        self.initial_point = self._linear_adjoint(y)

    def get_gradient(self, z, y):
        return self.linear_op.gramian(z) - self._linear_adjoint(y)

    def forward_module(self, x, y):
        return x - torch.exp(self.eta) * self.get_gradient(x, y)

    def forward(self, xk, y):
        if self.sigma == 0:
            for i in range(self.maxLayers):
                xk = self.forward_module(xk, y) - torch.exp(self.eta) * self.R(xk)
        else:
            self.noise_list = []
            for i in range(self.maxLayers):
                grad_step = self.forward_module(xk, y)
                self.noise_list.append((torch.max(grad_step) * self.sigma * torch.randn_like(grad_step)).to(xk.device))
                xk = self.forward_module(xk, y) - torch.exp(self.eta) * (self.R(xk) + self.noise_list[-1])
        return xk

class MRI_prox(nn.Module):
    def __init__(self, linear_op, DnCNN, args):
        super(MRI_prox, self).__init__()
        self.eta = nn.Parameter(torch.ones(1) * args.eta)
        self.linear_op = linear_op
        self.R = DnCNN
        self.maxLayers = args.maxiters
        self.sigma = args.sigma_wk if args.train_mode == 'sgd-jitter' and args.train else 0
    def _linear_op(self, x):
        return self.linear_op.forward(x)

    def _linear_adjoint(self, x):
        return self.linear_op.adjoint(x)

    def set_initial_point(self, y):
        self.initial_point = self._linear_adjoint(y)

    def get_gradient(self, z, y):
        return self.linear_op.gramian(z) - self._linear_adjoint(y)

    def forward_module(self, x, y):
        return x - torch.exp(self.eta) * self.get_gradient(x, y)

    def forward(self, xk, y):
        if self.sigma == 0:
            for i in range(self.maxLayers):
                xk = self.R(self.forward_module(xk, y))
        else:
            self.noise_list = []
            for i in range(self.maxLayers):
                grad_step = self.forward_module(xk, y)
                self.noise_list.append((torch.max(grad_step) * self.sigma * torch.randn_like(grad_step)).to(xk.device))
                xk = self.R(self.forward_module(xk, y) - torch.exp(self.eta) * self.noise_list[-1])
        return xk