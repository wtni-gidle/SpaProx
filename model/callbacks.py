import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import datetime
from matplotlib import pyplot as plt
import torch


# class LossHistory():
#     def __init__(self, log_dir, num_epochs, early_stopping = False, **kwargs):
#         curr_time = datetime.datetime.now()
#         time_str = datetime.datetime.strftime(curr_time, "%Y_%m_%d_%H_%M_%S")
#         self.log_dir = log_dir
#         self.time_str = time_str
#         self.save_path = os.path.join(self.log_dir, "loss_" + str(self.time_str))
#         self.train_loss = []
#         if early_stopping:
#             self.val_loss = []
#             self.estp = EarlyStopping(path = self.save_path, **kwargs)

#         self.curr_epoch = 0
#         self.num_epochs = num_epochs
#         os.makedirs(self.save_path)

#     def add_loss(self, train_loss, val_loss = None):
#         self.curr_epoch += 1

#         self.train_loss.append(train_loss)
#         with open(os.path.join(self.save_path, "epoch_train_loss_" + str(self.time_str) + ".txt"), "a") as f:
#             f.write(str(train_loss))
#             f.write("\n")

#         if hasattr(self, "val_loss"):
#             self.val_loss.append(val_loss)
#             with open(os.path.join(self.save_path, "epoch_val_loss_" + str(self.time_str) + ".txt"), "a") as f:
#                 f.write(str(val_loss))
#                 f.write("\n")
#         self.loss_plot()
    
#     def save_model(self, model, train_loss, val_loss = None, return_path = True):
#         if val_loss is None:
#             path = os.path.join(self.save_path, "ep%03d-train_loss%.4f.pth" % (self.num_epochs, train_loss))
#         else:
#             path = os.path.join(self.save_path, "ep%03d-train_loss%.4f-val_loss%.4f.pth" % (self.num_epochs, train_loss, val_loss))
            
#         torch.save(model.state_dict(), path)
#         if return_path:
#             return path

#     def loss_process(self, model, train_loss, val_loss = None):
#         self.add_loss(train_loss, val_loss)
#         path = self.save_model(model, train_loss, val_loss = None)
#         if hasattr(self, "estp"):
#             self.estp(val_loss, path)

#     def loss_plot(self):
#         iters = np.arange(len(self.train_loss))
#         with plt.style.context(["seaborn-bright"]):

#             plt.figure()
#             plt.plot(iters, self.train_loss, linewidth = 2, label = "train_loss")
            
#             if hasattr(self, "val_loss"):
#                 plt.plot(iters, self.val_loss, linewidth = 2, label = "val_loss")

#             plt.grid(True)
#             plt.xlabel("Epoch")
#             plt.ylabel("Loss")
#             plt.xticks(ticks = iters, labels = iters + 1)
#             plt.locator_params("x", nbins = 10)
#             plt.legend(loc = "upper right")

#             plt.savefig(os.path.join(self.save_path, "epoch_loss_" + str(self.time_str) + ".png"))

#             plt.cla()
#             plt.close("all")


class EarlyStopping:
    """Early stops the training if validation loss doesn"t improve after a given patience."""
    def __init__(self, patience = 7, delta = 0, verbose = True, trace_func = print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: True
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: "checkpoint.pt"
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func

    def __call__(self, val_loss, path):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.checkpoint(val_loss, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping")
        else:
            self.best_score = score
            self.checkpoint(val_loss, path)
            self.counter = 0

    def checkpoint(self, val_loss, path):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f})")
        self.val_loss_min = val_loss
        self.best_path = path
    

class MetricStorage():
    def __init__(self, log_dir, train_only = False):
        curr_time = datetime.datetime.now()
        self.time_str = datetime.datetime.strftime(curr_time, "%Y_%m_%d_%H_%M_%S")
        self.save_path = os.path.join(log_dir, "metrics_" + str(self.time_str))
        os.makedirs(self.save_path)
        
        self.train_loss = []
        self.train_only = train_only
        if not self.train_only:
            self.eval_metrics = {}

        self.curr_epoch = 0

    def update(self, model, train_loss, eval_metrics = None):
        self.add(train_loss, eval_metrics)
        self.loss_plot()
        path = self.save_model(model)

        return path
    
    def add(self, train_loss, eval_metrics = None):
        self.curr_epoch += 1

        self.train_loss.append(train_loss)
        with open(os.path.join(self.save_path, "epoch_train_loss.txt"), "a") as f:
            f.write("%.4f" % train_loss)
            f.write("\n")
            
        if not self.train_only:
            if self.eval_metrics:
                self.eval_metrics = {k: [v] for k, v in eval_metrics}
            else:
                for name in self.eval_metrics:
                    self.eval_metrics[name] += [eval_metrics[name]]
            
            for k, v in eval_metrics:
                with open(os.path.join(self.save_path, f"epoch_val_{k}.txt"), "a") as f:
                    f.write("%.4f" % v)
                    f.write("\n")

    def save_model(self, model, return_path = True):
        if self.train_only:
            path = os.path.join(self.save_path, "ep%02d-train_loss%.4f.pth" % (self.curr_epoch, self.train_loss[-1]))
        else:
            path = os.path.join(self.save_path, "ep%02d-train_loss%.4f-val_loss%.4f.pth" % (self.curr_epoch, self.train_loss[-1], self.eval_metrics["loss"][-1]))
            
        torch.save(model.state_dict(), path)

        if return_path:
            return path
        
    def loss_plot(self):
        iters = np.arange(len(self.train_loss))
        with plt.style.context(["seaborn-bright"]):

            plt.figure()
            plt.plot(iters, self.train_loss, linewidth = 2, label = "train_loss")
            
            if not self.train_only:
                plt.plot(iters, self.eval_metrics["loss"], linewidth = 2, label = "val_loss")

            plt.grid(True)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.xticks(ticks = iters, labels = iters + 1)
            plt.locator_params("x", nbins = 10)
            plt.legend(loc = "upper right")

            plt.savefig(os.path.join(self.save_path, "epoch_loss_" + str(self.time_str) + ".png"))

            plt.cla()
            plt.close("all")