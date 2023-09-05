import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from tqdm import tqdm

from .utils import Accumulator
from .models import MLP
from imb_torch2 import get_device
from .callbacks import MetricStorage, EarlyStopping
from .training_args import TrainingArguments

class NNClassifier():
    def __init__(
        self, 
        args: TrainingArguments = None,
        num_features = 400,
        num_classes = 2,
        hidden_sizes = [400, 60],
        dropout_rate = None,
        max_epoch = 30,
        validation = True,
        early_stopping = False,
        metrics_dict = None,
        gpu_id = -1,
        log_dir = "logs"
    ):
        self.args = TrainingArguments() if args is None else args
        self.max_epoch = max_epoch
        self.validation = validation
        self.early_stopping = early_stopping
        self.metrics_dict = metrics_dict

        self.device = get_device(gpu_id)

        hidden_sizes = [num_features] + list(hidden_sizes)
        self.model = MLP(hidden_sizes, num_classes = num_classes, dropout_rate = dropout_rate)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr = self.args.learning_rate,
            betas = (0.9, 0.999),
            eps = 1e-08,
            weight_decay = 0,
            amsgrad = False
        )
        # self.loss_history = LossHistory(log_dir = "logs", max_epoch = max_iter, early_stopping = self.early_stopping, 
        #                            patience = patience, verbose = verbose)

        if self.early_stopping:
            self.estp = EarlyStopping(
                patience = self.args.estp_patience, 
                delta = self.args.estp_delta, 
                verbose = True
            )
            assert self.validation, "validation must be True if using early stopping"
        
        self.metric_storage = MetricStorage(log_dir = log_dir, train_only = not self.validation)
        
    def fit(self, train_dataset, eval_dataset = None):
        self.model.to(self.device)
        self.model.loss_fn.to(self.device)

        if self.validation and eval_dataset is None:
            train_dataset, eval_dataset = data.random_split(
                train_dataset, 
                [1 - self.args.eval_ratio, self.args.eval_ratio]
            )
        
        train_iter = self._load_data(train_dataset, self.args.train_batch_size, self.args.num_workers)

        if self.validation:
            eval_iter = self._load_data(eval_dataset, self.args.eval_batch_size, self.args.num_workers, False)

        for self.curr_epoch in range(self.max_epoch):
            eval_metrics = None
            train_loss = self._fit_one_epoch(train_iter)
            if self.validation:
                eval_metrics = self._eval_one_epoch(eval_iter, metrics_dict = self.metrics_dict)

            path = self.metric_storage.update(self.model, train_loss, eval_metrics)

            if self.early_stopping:
                self.estp(eval_metrics["loss"], path)
                if self.estp.early_stop:
                    break
            print("----------------------------------------------------------------")

        self.n_iter_ = self.curr_epoch + 1
        if self.early_stopping:
            best_path = self.estp.best_path
            self.load_model(best_path)

        return self
    
    def metric_summary(self):
        return [self.metric_storage.train_loss, self.metric_storage.eval_metrics]
    
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
    
    def _load_data(self, dataset, batch_size, num_workers, shuffle = True):
        """
        Split data and convert data to the input form of the network.

        `np.ndarray` -> `DataLoader`
        """
        # dataset = LabeledDataset(ldp) if labeled else UnlabeledDataset(ldp)

        data_iter = data.DataLoader(
            dataset, 
            batch_size = batch_size, 
            shuffle = shuffle,
            num_workers = num_workers,
            pin_memory = False
        )

        return data_iter
    
    def _fit_one_epoch(self, train_iter):
        metric_train = Accumulator(2)
        train_num_steps = len(train_iter)

        if isinstance(self.model, nn.Module):
            self.model.train()

        with tqdm(total = train_num_steps, desc = f"Epoch [{self.curr_epoch + 1}/{self.max_epoch}]") as pbar:
            for x, y in train_iter:
                x = x.to(self.device)
                y = y.to(self.device)
                y_hat = self.model(x)
                loss = self.model.loss_fn(y_hat, y)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                metric_train.add(
                    float(loss * len(y)),
                    len(y)
                )
                pbar.update(1)

            train_loss = metric_train[0] / metric_train[1]
            pbar.set_postfix({"train_loss": "%.4f" % train_loss})

        return train_loss

    def _eval_one_epoch(self, eval_iter, metrics_dict: dict = None):
        #metric_dict每个值是一个函数，接受pred和true
        if isinstance(self.model, nn.Module):
            self.model.eval()

        with torch.no_grad():
            pbar = tqdm(total = len(eval_iter), desc = "Evaluating")
            result = [(self.model(x), y, pbar.update(1)) for x, y in eval_iter]
            # result = list(tqdm([(self.model(x), y) for x, y in eval_iter], desc = "Evaluating"))
            y_hat, y_true, _ = zip(*result)
            y_hat = torch.cat(y_hat)
            y_true = torch.cat(y_true)
            
            # loss
            eval_loss = self.model.loss_fn(y_hat, y_true).item()
            pbar.set_postfix({"val_loss": "%.4f" % eval_loss})
            pbar.close()
            # 转化为概率，后续计算
            y_pred = F.softmax(y_hat, dim = 1)
            y_pred = y_pred.to(torch.device("cpu")).numpy()
            y_true = y_true.to(torch.device("cpu")).numpy()

        if metrics_dict is not None:
            metrics = {name: fn(y_pred, y_true) for name, fn in metrics_dict.items()}
            metrics["loss"] = eval_loss
        else:
            metrics = {"loss": eval_loss}

        return metrics

    def predict(self, test_dataset, labeled = True):
        """
        Forward computation.

        `np.ndarray` -> `torch.tensor`
        """
        test_iter = self._load_data(test_dataset, self.args.eval_batch_size, self.args.num_workers, shuffle = False)

        if isinstance(self.model, nn.Module):
            self.model.eval()

        with torch.no_grad():
            if labeled:
                y_true = torch.cat([y for _, y in test_iter])
                y_pred = torch.cat([F.softmax(self.model(x), dim = 1) for x, _ in test_iter])
                y_true = y_true.to(torch.device("cpu")).numpy()
                y_pred = y_pred.to(torch.device("cpu")).numpy()

                return y_pred, y_true
            else:
                y_pred = torch.cat([F.softmax(self.model(x), dim = 1) for [x] in test_iter])
                y_pred = y_pred.to(torch.device("cpu")).numpy()
                
                return y_pred
