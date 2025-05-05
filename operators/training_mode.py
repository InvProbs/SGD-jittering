import torch.nn as nn
import torch


def norms(Z, dim=1):
    if dim == 1:
        return Z.view(Z.shape[0], -1).norm(dim=1)[:, None]
    elif dim == 2:
        return Z.view(Z.shape[0], -1).norm(dim=1)[:, None, None]
    elif dim == 3:
        return Z.view(Z.shape[0], -1).norm(dim=1)[:, None, None, None]
    else:
        return None

def norms_2D(Z):
    return Z.view(Z.shape[0], -1).norm(dim=1)[:, None, None]

def norms_3D(Z):
    return Z.view(Z.shape[0], -1).norm(dim=1)[:, None, None, None]

def PGD(net, X, X0, y, epsilon, alpha, num_iter, dim=1, eps_mode='l2'):
    """modified from https://adversarial-ml-tutorial.org/adversarial_examples/"""
    delta = torch.zeros_like(y, requires_grad=True)
    loss_list = []
    if eps_mode == 'l2':
        for t in range(num_iter):
            loss = nn.MSELoss()(net(X0 + delta, y + delta), X)
            loss_list.append(loss.item())
            loss.backward()
            delta.data += alpha * delta.grad.detach() / norms(delta.grad.detach(), dim=dim)
            # delta.data = torch.min(torch.max(delta.detach(), -X), 1 - X)  # clip X+delta to [0,1]
            delta.data *= epsilon / norms(delta.detach(), dim=dim).clamp(min=epsilon)
            delta.grad.zero_()
    elif eps_mode == 'inf':
        for t in range(num_iter):
            loss = nn.MSELoss()(net(X0 + delta, y + delta), X)
            loss.backward()
            delta.data = (delta + alpha * delta.grad.detach().sign()).clamp(-epsilon, epsilon)
            delta.grad.zero_()
    return delta.detach(), loss_list


def PGD_v2(net, X, forward_op, y, epsilon, alpha, num_iter, dim=2, eps_mode='l2'):
    """https://adversarial-ml-tutorial.org/adversarial_examples/"""
    delta = torch.zeros_like(y, requires_grad=True)
    loss_list = []
    if eps_mode == 'l2':
        for t in range(num_iter):
            loss = nn.MSELoss()(net(forward_op.adjoint(y + delta), y + delta), X)
            loss_list.append(loss.item())
            loss.backward()
            delta.data += alpha * delta.grad.detach() / norms(delta.grad.detach(), dim=dim)
            # delta.data = torch.min(torch.max(delta.detach(), -X), 1 - X)  # clip X+delta to [0,1]
            delta.data *= epsilon / norms(delta.detach(), dim=dim).clamp(min=epsilon)
            delta.grad.zero_()
    elif eps_mode == 'inf':
        for t in range(num_iter):
            loss = nn.MSELoss()(net(forward_op.adjoint(y + delta), y + delta), X)
            loss.backward()
            delta.data = (delta + alpha * delta.grad.detach().sign()).clamp(-epsilon, epsilon)
            delta.grad.zero_()
    return delta.detach(), loss_list


def adv_attack_to_diffusion_model(model, x_t, t, y, forward_op, eps=0.05, steps=10, alpha=0.01):
    x_t = x_t.clone().detach().requires_grad_(True)
    delta = torch.zeros_like(x_t, requires_grad=True)
    # zeros = torch.zeros_like(y).to('cuda')
    zeros = torch.zeros((x_t.shape[0], 320, 320, 1)).to('cuda')
    for _ in range(steps):
        X0 = forward_op.adjoint(torch.cat((delta.permute(0,2,3,1), zeros),3) + y)[:, 0:1]
        # Predict noise
        noise_pred = model(x_t + delta, t, X0)

        # Define adversarial loss (e.g., L2 distance from true noise or target)
        loss = -torch.mean(noise_pred ** 2)  # Try to maximize model error

        # Compute gradient
        loss.backward()

        # Update delta
        delta.data += alpha * delta.grad.detach() / norms(delta.grad.detach(), dim=3)
        delta.data *= eps / norms(delta.detach(), dim=3).clamp(min=eps)
        # grad_sign = delta.grad.detach().sign()
        # delta.data = delta.data + alpha * grad_sign
        #
        # # Project back to epsilon-ball
        # delta.data = torch.clamp(delta.data, -eps, eps)

        delta.grad.zero_()

    x_adv = x_t + delta.detach()
    return x_adv.detach(), delta.detach()

def PGD_mri(net, X, forward_op, y, epsilon, alpha, num_iter, dim=2, eps_mode='l2'):
    """https://adversarial-ml-tutorial.org/adversarial_examples/"""
    delta = torch.zeros_like(y, requires_grad=True)
    loss_list = []
    if eps_mode == 'l2':
        for t in range(num_iter):
            loss = nn.MSELoss()(net(forward_op.adjoint(y + delta)), X)
            loss_list.append(loss.item())
            loss.backward()
            delta.data += alpha * delta.grad.detach() / norms(delta.grad.detach(), dim=dim)
            # delta.data = torch.min(torch.max(delta.detach(), -X), 1 - X)  # clip X+delta to [0,1]
            delta.data *= epsilon / norms(delta.detach(), dim=dim).clamp(min=epsilon)
            delta.grad.zero_()
    elif eps_mode == 'inf':
        for t in range(num_iter):
            loss = nn.MSELoss()(net(forward_op.adjoint(y + delta)), X)
            loss.backward()
            delta.data = (delta + alpha * delta.grad.detach().sign()).clamp(-epsilon, epsilon)
            delta.grad.zero_()
    return delta.detach(), loss_list
