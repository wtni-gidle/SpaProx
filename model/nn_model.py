from typing import Optional
import numpy as np
from collections import OrderedDict
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch import Tensor
from tqdm import tqdm

from model.utils import DeviceDataLoader, Accumulator
from model.callbacks import LossHistory
from model.losses import loss_func
from model.models import MLP



# class GHMCLoss(nn.Module):
#     def __init__(self, bins = 30, momentum = 0.75, num_classes = 2):
#         super().__init__()
#         self.bins = bins
#         self.momentum = momentum
#         edges = torch.arange(bins + 1).float() / bins
#         self.register_buffer("edges", edges)
#         self.edges[-1] += 1e-6
#         manum = None
#         self.register_buffer("manum", manum)
#         self.num_classes = num_classes

#     def _custom_loss(self, input, target, weight):
#         loss = F.cross_entropy(input = input, target = target, reduction = "none")
#         loss = (loss * weight).mean()

#         return loss


#     def _custom_loss_grad(self, input, target):
#         target = F.one_hot(target, num_classes = self.num_classes).detach()
#         y = F.softmax(input, dim = 1).detach()

#         return y - target
    
#     def forward(self, input, target):
#         # target一维
#         # grad: 梯度
#         grad = self._custom_loss_grad(input, target)/2.0
#         # g: 梯度向量的l1范数
#         g = torch.abs(grad).sum(dim = 1).view(-1, 1)
#         # 批量样本的梯度范数g在bins上的落点，BxN
#         edges = self.edges.view(1, -1)
#         g_bin = torch.logical_and(torch.ge(g, edges[:, :-1]), torch.less(g, edges[:, 1:]))
#         # 批量样本的梯度范数g归属bins的index，(B, )
#         bin_idx = torch.where(g_bin)[1]
#         # 每个bins的counts，(N, )
#         bin_count = torch.sum(g_bin, dim = 0)

#         N = len(target)

#         if self.momentum > 0:
#             if self.manum is None:
#                 self.manum = bin_count
#             else:
#                 self.manum = self.momentum * self.manum + (1 - self.momentum) * bin_count
        
#         else:
#             self.manum = bin_count
        
#         weight = N / (self.manum[bin_idx] * self.bins)


#         # nonempty_bins = (bin_count > 0).sum().item()

#         # gd = bin_count * nonempty_bins
#         # gd = torch.clamp(gd, min=0.0001)
#         # beta = N / gd

#         return self._custom_loss(input, target, weight)




# class FocalLoss(nn.Module):
#     def __init__(self, alpha = [0.5, 0.5], gamma = 2, num_classes = 2, reduction = "mean") -> None:
#         super().__init__()
#         self.reduction = reduction

#         if isinstance(alpha, list):
#             assert len(alpha) == num_classes
#             self.alpha = torch.Tensor(alpha)
#         else:
#             assert alpha < 1
#             self.alpha = torch.zeros(num_classes)
#             self.alpha[0] += alpha
#             self.alpha[1:] += (1 - alpha)

#         self.gamma = gamma

#     def forward(self, input: Tensor, target: Tensor) -> Tensor:
#         self.alpha = self.alpha.to(input.device)
#         preds_logsoft = F.log_softmax(input, dim = 1)
#         preds_softmax = torch.exp(preds_logsoft)

#         preds_softmax = preds_softmax.gather(1, target.view(-1, 1))
#         preds_logsoft = preds_logsoft.gather(1, target.view(-1, 1))
#         self.alpha = self.alpha.gather(0, target.view(-1))
#         loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma), preds_logsoft)

#         loss = torch.mul(self.alpha, loss.t())
#         if self.reduction == "mean":
#             loss = loss.mean()
#         elif self.reduction == "sum":
#             loss = loss.sum()
#         elif self.reduction == "none":
#             pass
#         else:
#             raise ValueError("{} is not a valid value for reduction".format(self.reduction))

#         return loss



# def loss_func(loss = "focal", **kwargs):
#     if loss == "focal":
#         return FocalLoss(**kwargs)
#     if loss == "cross_entropy":
#         return nn.CrossEntropyLoss(**kwargs)
#     if loss == "ghm":
#         return GHMCLoss(**kwargs)
#     else:
#         raise ValueError("{} is not a valid value for loss".format(loss))



class Net(nn.Module):
    """
    Two-layer feed-forward neural network, using Dropout.
    """
    
    def __init__(self, features: list, dropout_rate):
        super().__init__()
        self.hidden = self.make_layer(features[:-1], dropout_rate)
        self.output = nn.Sequential(OrderedDict([
            ("FC_{}".format(len(features) - 1), nn.Linear(features[-2], features[-1]))
            ])
        )
    
    def make_layer(self, layer, dropout_rate):
        layers = []

        for i in range(len(layer) - 1):
            layers.append((
                "FC_{}".format(i + 1), 
                nn.Linear(layer[i], layer[i + 1])
            ))
            layers.append((
                "ReLU_{}".format(i + 1),
                nn.ReLU()
            ))
            layers.append((
                "Dropout_{}".format(i + 1),
                nn.Dropout(p = dropout_rate)
            ))

        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        x = self.hidden(x)
        x = self.output(x)

        return x



def fit_one_epoch(model, loss_func, loss_history, optimizer, epoch, max_epoch, train_iter, verbose = True):
    metric_train = Accumulator(2)
    train_step = len(train_iter)

    if verbose:
        with tqdm(total = train_step, desc = f"Epoch [{epoch + 1}/{max_epoch}]") as pbar:

            if isinstance(model, nn.Module):
                model.train()
                
            for x, y in train_iter:
                y_hat = model(x)
                loss = loss_func(y_hat, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                metric_train.add(
                    float(loss * len(y)),
                    len(y)
                )
                pbar.update(1)

            train_loss = metric_train[0] / metric_train[1]
            pbar.set_postfix({"train_loss": round(train_loss, 4)})
    
    else:
        if isinstance(model, nn.Module):
                model.train()
                
        for X, y in train_iter:
            y_hat = model(X)
            loss = loss_func(y_hat, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            metric_train.add(
                float(loss * len(y)),
                len(y)
            )

        train_loss = metric_train[0] / metric_train[1]

    loss_history.loss_process(model, train_loss)



class NeuralNetworkClassifier(BaseEstimator, ClassifierMixin):
    """
    Build a Neural Network Classifier by inheriting `BaseEstimator` and 
    `ClassifierMixin`, in order to use sklearn API.
    """

    def __init__(
        self, 
        num_features: Optional[int] = None,
        hidden_layer_sizes = [400, 60],
        dropout_rate = 0.5,
        learning_rate = 0.001,
        max_iter = 50,
        batch_size = 16,
        valid_frac = 0.1,
        early_stopping = True,
        patience = 7,
        gpu_id = -1,
        verbose = True
    ):
        self.num_features = num_features
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.valid_frac = valid_frac
        self.early_stopping = early_stopping
        self.patience = patience
        self.verbose = verbose
        self.gpu_id = gpu_id

    # def fit(self, X, y, marked_x, marked_y):
    #     if self.num_features:
    #         start = int(X.shape[1] / 2)
    #         X = np.concatenate(
    #             (X[:, :self.num_features], X[:, start:(start + self.num_features)]),
    #             1
    #         )

    #     self.classes_, y = np.unique(y, return_inverse=True)

    #     input_size = X.shape[1]
    #     output_size = len(self.classes_)
    #     layer_sizes = [input_size] + list(self.hidden_layer_sizes) + [output_size]
    #     self.net_ = Net(features = layer_sizes, dropout_rate = self.dropout_rate).to(device)

    #     self.optimizer_ = torch.optim.Adam(
    #         self.net_.parameters(),
    #         lr = self.learning_rate,
    #         betas = (0.9, 0.999),
    #         eps = 1e-08,
    #         weight_decay = 0,
    #         amsgrad = False
    #     )
        

    #     self.loss_func_ = loss_func("ce").to(device)

    #     train_iter, val_iter = self._preprocess(X, y)

    #     loss_history = LossHistory(log_dir = "logs", max_epoch = self.max_iter, early_stopping = self.early_stopping, 
    #                                patience = self.patience, verbose = self.verbose)

    #     for epoch in range(10):
    #         fit_one_epoch(self.net_, self.loss_func_, loss_history, self.optimizer_, epoch, self.max_iter, 
    #                       train_iter, val_iter, verbose = self.verbose)
    #         if self.verbose:
    #             print("----------------------------------------------------------------")
    #         if self.early_stopping and loss_history.estp.early_stop:
    #             break

    #     self.n_iter_ = epoch + 1
    #     if self.early_stopping:
    #         path = loss_history.estp.get()
    #         self.net_.load_state_dict(torch.load(path))
        

    #     self.loss_func_ = loss_func("ghm").to(device)
    #     X = np.concatenate((X, marked_x))
    #     y = np.concatenate((y, marked_y))
    #     train_iter, val_iter = self._preprocess(X, y)
    #     loss_history = LossHistory(log_dir = "logs", max_epoch = self.max_iter, early_stopping = self.early_stopping, 
    #                                patience = self.patience, verbose = self.verbose)
    #     for epoch in range(self.max_iter):
    #         fit_one_epoch(self.net_, self.loss_func_, loss_history, self.optimizer_, epoch, self.max_iter, 
    #                       train_iter, val_iter, verbose = self.verbose)
    #         if self.verbose:
    #             print("----------------------------------------------------------------")
    #         if self.early_stopping and loss_history.estp.early_stop:
    #             break
    #     if self.early_stopping:
    #         path = loss_history.estp.get()
    #         self.net_.load_state_dict(torch.load(path))

    #     return self
    def get_device(self):
        if self.gpu_id == -1:
            return torch.device("cpu")
        else:
            return torch.device("cuda:" + str(self.gpu_id) if torch.cuda.is_available() else "cpu")
        

    def fit(self, X, y):
        self.device = self.get_device()
        if self.num_features:
            start = int(X.shape[1] / 2)
            X = np.concatenate(
                (X[:, :self.num_features], X[:, start:(start + self.num_features)]),
                1
            )

        self.classes_, y = np.unique(y, return_inverse=True)

        input_size = X.shape[1]
        output_size = len(self.classes_)
        layer_sizes = [input_size] + list(self.hidden_layer_sizes) + [output_size]
        self.net_ = Net(features = layer_sizes, dropout_rate = self.dropout_rate).to(self.device)

        self.optimizer_ = torch.optim.Adam(
            self.net_.parameters(),
            lr = self.learning_rate,
            betas = (0.9, 0.999),
            eps = 1e-08,
            weight_decay = 0,
            amsgrad = False
        )
        

        self.loss_func_ = loss_func("ghm").to(self.device)

        train_iter, val_iter = self._preprocess(X, y)

        loss_history = LossHistory(log_dir = "logs", max_epoch = self.max_iter, early_stopping = self.early_stopping, 
                                   patience = self.patience, verbose = self.verbose)

        for epoch in range(self.max_iter):
            fit_one_epoch(self.net_, self.loss_func_, loss_history, self.optimizer_, epoch, self.max_iter, 
                          train_iter, val_iter, verbose = self.verbose)
            if self.verbose:
                print("----------------------------------------------------------------")
            if self.early_stopping and loss_history.estp.early_stop:
                break

        self.n_iter_ = epoch + 1
        if self.early_stopping:
            path = loss_history.estp.get()
            self.net_.load_state_dict(torch.load(path))

        return self

    def _preprocess(self, X, y):
        """
        Split data and convert data to the input form of the network.

        `np.ndarray` -> `DataLoader`
        """

        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size = self.valid_frac,
            stratify = y
        )

        # region train_iter
        X_train = torch.tensor(X_train, dtype = torch.float)
        y_train = torch.tensor(y_train, dtype = torch.long)

        train_dataset = data.TensorDataset(X_train, y_train)
        train_iter = data.DataLoader(
            train_dataset, 
            batch_size = self.batch_size, 
            shuffle = True
        )
        train_iter = DeviceDataLoader(train_iter, self.device)
        # endregion

        # region val_iter
        X_val = torch.tensor(X_val, dtype = torch.float)
        y_val = torch.tensor(y_val, dtype = torch.long)

        val_dataset = data.TensorDataset(X_val, y_val)
        val_iter = data.DataLoader(
            val_dataset, 
            batch_size = self.batch_size, 
            shuffle = False
        )
        val_iter = DeviceDataLoader(val_iter, self.device)
        # endregion

        return train_iter, val_iter

    def _forward_out(self, X, batchsize = 128):
        """
        Forward computation.

        `np.ndarray` -> `torch.tensor`
        """

        # region test_iter
        X_test = torch.tensor(X, dtype = torch.float)

        test_dataset = data.TensorDataset(X_test)
        test_iter = data.DataLoader(
            test_dataset, 
            batch_size = batchsize, 
            shuffle = False
        )
        test_iter = DeviceDataLoader(test_iter, self.device)
        # endregion

        if isinstance(self.net_, nn.Module):
            self.net_.eval()

        output = torch.tensor([]).to(self.device)
        with torch.no_grad():
            for [x] in test_iter:
                out = self.net_(x)
                out = F.softmax(out, dim = 1)
                output = torch.cat((output, out))
        
        return output
    

    # def test_forward(self, X, y, batchsize = 128):
    #     """
    #     Forward computation.

    #     `np.ndarray` -> `torch.tensor`
    #     """
    #     # region test_iter
    #     X_test = torch.tensor(X, dtype = torch.float)
    #     y_test = torch.tensor(y, dtype = torch.long)
    #     test_dataset = data.TensorDataset(X_test, y_test)
    #     test_iter = data.DataLoader(
    #         test_dataset, 
    #         batch_size = batchsize, 
    #         shuffle = False
    #     )
    #     test_iter = DeviceDataLoader(test_iter, device)
    #     # endregion

    #     if isinstance(self.net_, nn.Module):
    #         self.net_.eval()

    #     output = torch.tensor([]).to(device)
    #     with torch.no_grad():
    #         for x, _ in test_iter:
    #             out = self.net_(x)
    #             # out = F.softmax(out, dim = 1)
    #             output = torch.cat((output, out))
    #     y_test = y_test.to(device)
    #     return output, y_test

    def predict(self, X, batchsize = 256):
        """
        Predict.
        """
        if self.num_features:
            start = int(X.shape[1] / 2)
            X = np.concatenate(
                (X[:, :self.num_features], X[:, start:(start + self.num_features)]),
                1
            )
        output = self._forward_out(X, batchsize = batchsize)

        output = output.to(torch.device("cpu")).numpy()
        pred = self.classes_[np.argmax(output, axis = 1)]
        
        return pred

    def predict_proba(self, X, batchsize = 256):
        """
        Predict probability.
        """
        if self.num_features:
            start = int(X.shape[1] / 2)
            X = np.concatenate(
                (X[:, :self.num_features], X[:, start:(start + self.num_features)]),
                1
            )
        output = self._forward_out(X, batchsize = batchsize)

        output = output.to(torch.device("cpu")).numpy()

        return output


