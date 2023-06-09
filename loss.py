from typing import Optional
import numpy as np
import imb
import pickle
import datapre as DP
import cupy as cp
import torch
import evaluation as eval
from model.callbacks import LossHistory
from model.nn_model import *

class MyModel(NeuralNetworkClassifier):
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

    def fit(self, X, y, marked_x, marked_y):
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
        self.net_ = Net(features = layer_sizes, dropout_rate = self.dropout_rate).to(device)

        self.optimizer_ = torch.optim.Adam(
            self.net_.parameters(),
            lr = self.learning_rate,
            betas = (0.9, 0.999),
            eps = 1e-08,
            weight_decay = 0,
            amsgrad = False
        )
        

        self.loss_func_ = loss_func("cross_entropy").to(device)

        train_iter, val_iter = self._preprocess(X, y)

        loss_history = LossHistory(log_dir = "logs", max_epoch = self.max_iter, early_stopping = self.early_stopping, 
                                   patience = self.patience, verbose = self.verbose)

        for epoch in range(10):
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
        

        self.loss_func_ = loss_func("ghm").to(device)
        X = np.concatenate((X, marked_x))
        y = np.concatenate((y, marked_y))
        train_iter, val_iter = self._preprocess(X, y)
        loss_history = LossHistory(log_dir = "logs", max_epoch = self.max_iter, early_stopping = self.early_stopping, 
                                   patience = self.patience, verbose = self.verbose)
        for epoch in range(self.max_iter):
            epoch += 10
            fit_one_epoch(self.net_, self.loss_func_, loss_history, self.optimizer_, epoch, self.max_iter, 
                          train_iter, val_iter, verbose = self.verbose)
            if self.verbose:
                print("----------------------------------------------------------------")
            if self.early_stopping and loss_history.estp.early_stop:
                break
        if self.early_stopping:
            path = loss_history.estp.get()
            self.net_.load_state_dict(torch.load(path))

        return self
    

if __name__ == "__main__":

    dirname = "mouse_brain_sagittal_anterior"
    GPU_ID = 1
    with open(dirname + "/train_data.pkl", "rb") as file:
        train_data = pickle.load(file)
    with open(dirname + "/test_data.pkl", "rb") as file:
        test_data = pickle.load(file)
    DP.setup_seed(1)
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    with cp.cuda.Device(GPU_ID):
        pos_index, neg_index, marked_neg_index = imb.eliminate_BD_neg(train_data.feature, train_data.label, k = 20)
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        marked_neg_index = cp.asnumpy(marked_neg_index)

    marked_feature = train_data.get_feature(train_data.pair_index_son[marked_neg_index], copy = True)
    marked_label = train_data.get_label(train_data.data_index[marked_neg_index], copy = True)
    train_data.pop(marked_neg_index)
    train_data.mirror_copy()
    train_data.get_feature()
    train_data.get_label()
    model = MyModel(batch_size = 128, verbose = False)
    model.fit(train_data.feature, train_data.label, marked_feature, marked_label)
    # train_data(无marked_neg_index)
    predprob = model.predict_proba(train_data.feature)
    r1 = eval.evaluate(train_data.label, predprob, verbose = False)
    # marked_neg_index
    predprob = model.predict_proba(marked_feature)
    r2 = eval.evaluate(marked_label, predprob, verbose = False)
    # 全部test_data
    predprob = model.predict_proba(test_data.feature)
    r3 = eval.evaluate(test_data.label, predprob, verbose = False)
    with cp.cuda.Device(GPU_ID):
        pos_index, neg_index, marked_neg_index_test = imb.eliminate_BD_neg(test_data.feature, test_data.label, k = 20)
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        marked_neg_index_test = cp.asnumpy(marked_neg_index_test)
    test_data.pop(marked_neg_index_test)
    test_data.mirror_copy()
    test_data.get_feature()
    test_data.get_label()
    # test_data(去除marked_neg_index)
    predprob = model.predict_proba(test_data.feature)
    r4 = eval.evaluate(test_data.label, predprob, verbose = True)
    with open(dirname + "/result3.pkl", "wb") as file:
        pickle.dump([r1,r2,r3,r4], file)



