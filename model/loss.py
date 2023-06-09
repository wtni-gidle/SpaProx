import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np


# todo
class GHMCLoss(nn.Module):
    def __init__(self, bins = 30, momentum = 0.75, num_class = 2, reduction = "mean"):
        super().__init__()
        self.bins = bins
        self.momentum = momentum
        self.edges = torch.arange(bins + 1).float() / bins
        self.edges[-1] += 1e-6
        if momentum > 0:
            self.acc_sum = torch.zeros(bins)
        self.num_class = num_class
        self.reduction = reduction

    def _custom_loss(self, input, target, weight):
        loss = F.cross_entropy(input = input, target = target, reduction = "none")
        loss = (loss * weight).mean()

        return loss

    def _custom_loss_grad(self, input, target):
        target = F.one_hot(target, num_classes = self.num_class)
        y = F.softmax(input, dim = 1).detach()

        return y - target
    
    def forward(self, input: Tensor, target: Tensor):
        edges = self.edges.to(input.device)
        mmt = self.momentum
        self.acc_sum = self.acc_sum.to(input.device)
        # target一维
        # grad: 梯度
        grad = self._custom_loss_grad(input, target)/2.0
        # g: 梯度向量的l1范数
        g = torch.abs(grad).sum(dim = 1).view(-1, 1)
        # 批量样本的梯度范数g在bins上的落点，BxN
        edges = edges.view(1, -1)
        g_bin = torch.logical_and(torch.ge(g, edges[:, :-1]), torch.less(g, edges[:, 1:]))
        # 批量样本的梯度范数g归属bins的index，(B, )
        bin_idx = torch.where(g_bin)[1]
        # 每个bins的counts，(N, )
        bin_count = torch.sum(g_bin, dim = 0, dtype = torch.float32)

        N = len(target)
        M = (bin_count > 0).sum().item()

        if mmt > 0:
            bin_count[bin_count == 0] = self.acc_sum[bin_count == 0]
            self.acc_sum = mmt * self.acc_sum + (1 - mmt) * bin_count
            weight = N / self.acc_sum[bin_idx]
        else:
            weight = N / bin_count[bin_idx]
        
        weight = weight / M


        return self._custom_loss(input, target, weight)




class FocalLoss(nn.Module):
    def __init__(self, alpha = None, gamma = 2, num_class = 2, reduction = "mean") -> None:
        super().__init__()
        if alpha is None:
            self.alpha = torch.ones(num_class)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            self.alpha = torch.FloatTensor(alpha).view(num_class, 1)
            self.alpha = self.alpha / self.alpha.sum()
        elif isinstance(self.alpha, float):
            assert alpha < 1
            self.alpha = torch.zeros(num_class)
            self.alpha[0] = alpha
            self.alpha[1:] = (1 - alpha)
        else:
            raise TypeError('Not support alpha type')
        
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        alpha = self.alpha.to(target.device)

        logpt = F.log_softmax(input, dim = 1)
        logpt = logpt.gather(1, target.view(-1, 1))
        logpt = logpt.view(-1)

        pt = torch.exp(logpt)

        alpha = alpha.gather(0, target.view(-1))

        loss = -1 * alpha * torch.pow((1 - pt), self.gamma) * logpt

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "none":
            pass
        else:
            raise ValueError("{} is not a valid value for reduction".format(self.reduction))

        return loss



def loss_func(loss = "fl", **kwargs):
    if loss == "fl":
        return FocalLoss(**kwargs)
    if loss == "ce":
        return nn.CrossEntropyLoss(**kwargs)
    if loss == "ghm":
        return GHMCLoss(**kwargs)
    else:
        raise ValueError("{} is not a valid value for loss".format(loss))
    

