from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from tqdm import tqdm

from model.utils import DeviceDataLoader, Accumulator
from model.callbacks import LossHistory
from model.models import MLP
from datasets.dataset_wrappers import LabeledDataset, UnlabeledDataset
from imb_torch2 import get_device
from .hooks import LossHook, HookBase, MetricHook
from weakref import proxy

def fit_one_epoch(model: nn.Module, loss_history, optimizer, epoch, max_epoch, train_iter):
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
                loss = model.loss_fn(y_hat, y)
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



class Trainer():
    def __init__(
        self, 
        args,#大部分参数配置
        model: nn.Module, 
        optimizer: torch.optim.Optimizer,
        lr_scheduler,
        train_dataset: data.Dataset,
        val_dataset: Union[data.Dataset, list[data.Dataset]],
        early_stopping,
        num_epochs,
        device: torch.device = torch.device("cuda:0")
    ) -> None:
        # args:大部分参数配置
        # val_dataset每个epoch都会评估，如果早停，则第一个数据集用作早停数据集
        self.args = args
        self.model = model
        self.optimizer = optimizer

        if val_dataset is None:
            assert early_stopping, "val_dataset cannot be None if using early_stopping"
        else:
            val_dataset = val_dataset if isinstance(val_dataset, list) else [val_dataset]
        
        self._load_data(train_dataset, val_dataset)

        self.device = device
        self.early_stopping = early_stopping
        self.num_epochs = num_epochs
        self.curr_epoch = 0
    
    def register_hooks(self):
        self.losshook = LossHook("losshook")
        self.losshook.trainer = proxy(self)
        

    def train(self):
        for epoch in range(self.max_iter):
            self._before_epoch()
            self.train_one_epoch()
            self._after_epoch()

    def _load_data(self, train_dataset, val_dataset = None):
        self.train_iter = data.DataLoader(
            train_dataset, 
            batch_size = self.args.train_batch_size, 
            shuffle = True,
            num_workers = self.args.num_workers,
            pin_memory = False
        )
        if val_dataset is not None:
            #data.random_split(self.train_dataset, [])
            ds2dl = lambda x: data.DataLoader(
                x, 
                batch_size = self.args.eval_batch_size, 
                shuffle = False,
                num_workers = self.args.num_workers,
                pin_memory = False
            )
            self.val_iter = [ds2dl(ds) for ds in val_dataset]

        

    def train_one_epoch(self):
        if isinstance(self.model, nn.Module):
            self.model.train()
                
        for x, y in self.train_iter:
            self._before_iter()
            self._train_one_iter(x, y)
            self._after_iter()
    
    def _train_one_iter(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)
        y_hat = self.model(x)
        loss = self.model.loss_fn(y_hat, y)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.iter_loss = loss
        self.iter_size = len(y)

    def _before_iter(self):
        pass

    def _after_iter(self):
        self.losshook.after_iter()

        self.epoch_pbar.update(1)

    def _before_epoch(self):
        self.losshook.before_epoch()
        self.train_num_steps = len(self.train_iter)
        self.epoch_pbar = tqdm(total = self.train_num_steps, desc = f"Epoch [{self.curr_epoch + 1}/{self.num_epochs}]")

    def _after_epoch(self):
        epoch_loss = self.losshook.after_epoch()
        self.epoch_pbar.set_postfix({"train_loss": "%.4f" % epoch_loss})
        self.epoch_pbar.close()

    