from typing import Optional
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch import Tensor
from tqdm import tqdm

from model.utils import DeviceDataLoader, Accumulator
from model.callbacks import LossHistory
from model.losses import loss_func
from models import MLP
from datasets.dataset_wrappers import LabeledDataset, UnlabeledDataset
from imb_torch2 import get_device


def fit_one_epoch(model: nn.Module, loss_func, loss_history, optimizer, epoch, max_epoch, train_iter, verbose = True):
    metric_train = Accumulator(2)
    train_step = len(train_iter)

    device = next(model.parameters()).device

    if verbose:
        with tqdm(total = train_step, desc = f"Epoch [{epoch + 1}/{max_epoch}]") as pbar:

            if isinstance(model, nn.Module):
                model.train()
                
            for x, y in train_iter:
                x = x.to(device)
                y = y.to(device)
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
            x = x.to(device)
            y = y.to(device)
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



class NNClassifier():
    def __init__(
        self, 
        num_features,
        num_classes = 2,
        hidden_sizes = [400, 60],
        dropout_rate = None,
        learning_rate = 0.001,
        max_iter = 50,
        batch_size = 16,
        early_stopping = False,
        patience = 7,
        gpu_id = -1,
        verbose = True
    ):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.verbose = verbose

        self.device = get_device(gpu_id)

        hidden_sizes = [num_features] + list(hidden_sizes)
        self.net_ = MLP(hidden_sizes, num_classes = num_classes, dropout_rate = dropout_rate)

        self.optimizer_ = torch.optim.Adam(
            self.net_.parameters(),
            lr = learning_rate,
            betas = (0.9, 0.999),
            eps = 1e-08,
            weight_decay = 0,
            amsgrad = False
        )
        self.loss_history = LossHistory(log_dir = "logs", max_epoch = max_iter, early_stopping = self.early_stopping, 
                                   patience = patience, verbose = verbose)
        
    def fit(self, train_ldp):
        self.net_.to(self.device)

        self.loss_func_ = loss_func("ce").to(self.device)

        train_iter = self._preprocess(train_ldp, batch_size = self.batch_size)

        for epoch in range(self.max_iter):
            fit_one_epoch(self.net_, self.loss_func_, self.loss_history, self.optimizer_, epoch, self.max_iter, 
                          train_iter, verbose = self.verbose)
            if self.verbose:
                print("----------------------------------------------------------------")
            if self.early_stopping and self.loss_history.estp.early_stop:
                break

        self.n_iter_ = epoch + 1
        if self.early_stopping:
            path = self.loss_history.estp.get()
            self.net_.load_state_dict(torch.load(path))

        return self

    def _preprocess(self, ldp, batch_size, labeled = True, shuffle = True):
        """
        Split data and convert data to the input form of the network.

        `np.ndarray` -> `DataLoader`
        """
        dataset = LabeledDataset(ldp) if labeled else UnlabeledDataset(ldp)

        data_iter = data.DataLoader(
            dataset, 
            batch_size = batch_size, 
            shuffle = shuffle
        )
        data_iter = DeviceDataLoader(data_iter, self.device)

        return data_iter

    def predict(self, test_ldp, batch_size = 128, labeled = True):
        """
        Forward computation.

        `np.ndarray` -> `torch.tensor`
        """
        test_iter = self._preprocess(test_ldp, batch_size, labeled, shuffle = False)

        if isinstance(self.net_, nn.Module):
            self.net_.eval()

        with torch.no_grad():
            if labeled:
                y_true = torch.cat([y for _, y in test_iter])
                y_pred = torch.cat([F.softmax(self.net_(x), dim = 1) for x, _ in test_iter])
                y_true = y_true.to(torch.device("cpu")).numpy()
                y_pred = y_pred.to(torch.device("cpu")).numpy()

                return y_pred, y_true
            else:
                y_pred = torch.cat([F.softmax(self.net_(x), dim = 1) for [x] in test_iter])
                y_pred = y_pred.to(torch.device("cpu")).numpy()
                
                return y_pred



