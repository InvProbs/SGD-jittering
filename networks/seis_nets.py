import torch.nn as nn
import torch


class single_layer(nn.Module):
    def __init__(self, args):
        super(single_layer, self).__init__()
        ks = 3
        pad = 1
        model = nn.Sequential(nn.Conv2d(2, 32, kernel_size=ks, padding=pad),
                              nn.GroupNorm(4, 32),
                              nn.ReLU(),
                              nn.Conv2d(32, 32, kernel_size=ks, padding=pad),
                              nn.GroupNorm(4, 32),
                              nn.ReLU(),
                              nn.Conv2d(32, 32, kernel_size=ks, padding=pad),
                              nn.GroupNorm(4, 32),
                              nn.ReLU(),
                              nn.Conv2d(32, 1, kernel_size=ks, padding=pad),
                              nn.Conv2d(1, 1, kernel_size=1),
                              )
        self.data_layer = nn.Sequential(*model)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            with torch.no_grad():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', a=0.001)
                    m.weight /= 10
                elif isinstance(m, nn.GroupNorm):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, y, xk):
        inp = self.data_layer(torch.cat((y, xk), dim=1))
        return inp




class seis_gd(nn.Module):
    def __init__(self, linear_op, DnCNN, args):
        super(seis_gd, self).__init__()
        self.eta = nn.Parameter(torch.ones(1) * args.eta)
        self.linear_op = linear_op
        self.R = DnCNN
        self.maxLayers = args.maxiters
        self.sigma = args.sigma_wk if args.train_mode == 'sgd-jitter' and args.train else 0
        self.sigmoid = nn.Sigmoid()
    def _linear_op(self, x):
        return self.linear_op.forward(x)

    def _linear_adjoint(self, x):
        return self.linear_op.adjoint(x)

    def get_gradient(self, z, y):
        return self.linear_op.gramian(z) - self._linear_adjoint(y)

    def forward_module(self, x, y):
        return x - 0.1 * self.sigmoid(self.eta) * self.get_gradient(x, y)

    def forward(self, xk, y):
        if self.sigma == 0:
            for i in range(self.maxLayers):
                xk = self.forward_module(xk, y) - 0.1 * self.sigmoid(self.eta) * self.R(y, xk)
        else:
            self.noise_list = []
            for i in range(self.maxLayers):
                grad_step = self.forward_module(xk, y)
                self.noise_list.append((torch.max(grad_step) * self.sigma * torch.randn_like(grad_step)).to(xk.device))
                xk = self.forward_module(xk, y) - 0.1 * self.sigmoid(self.eta) * (self.R(y, xk) + self.noise_list[-1])
        return xk

class seis_proxgd(nn.Module):
    def __init__(self, linear_op, DnCNN, args):
        super(seis_proxgd, self).__init__()
        self.eta = nn.Parameter(torch.ones(1) * args.eta)
        self.linear_op = linear_op
        self.R = DnCNN
        self.maxLayers = args.maxiters
        self.sigma = args.sigma_wk if args.train_mode == 'sgd-jitter' and args.train else 0
        self.sigmoid = nn.Sigmoid()
    def _linear_op(self, x):
        return self.linear_op.forward(x)

    def _linear_adjoint(self, x):
        return self.linear_op.adjoint(x)

    def get_gradient(self, z, y):
        return self.linear_op.gramian(z) - self._linear_adjoint(y)

    def forward_module(self, x, y):
        return x - 0.1 * self.sigmoid(self.eta) * self.get_gradient(x, y)
    def forward(self, xk, y):
        if self.sigma == 0:
            for i in range(self.maxLayers):
                xk = self.R(y,self.forward_module(xk, y))
        else:
            self.noise_list = []
            for i in range(self.maxLayers):
                grad_step = self.forward_module(xk, y)
                self.noise_list.append((torch.max(grad_step) * self.sigma * torch.randn_like(grad_step)).to(xk.device))
                xk = self.R(y, self.forward_module(xk, y) - torch.exp(self.eta) * self.noise_list[-1])
        return xk