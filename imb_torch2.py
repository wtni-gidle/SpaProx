from typing import Union, Tuple
import torch
import numpy as np
from sklearn.utils import gen_batches
from datasets.data_process import LabeledDataUnit, LabeledDataDoublet

class DataLoader():
    def __init__(
        self, 
        ldp: Union[LabeledDataUnit, LabeledDataDoublet], 
        data_index: Union[list, np.ndarray],
        batch_size: int = 32
    ) -> None:
        self.ldp = ldp
        self.data_index = data_index
        self.batch_size = batch_size
        self.num_samples = len(self.data_index)
    
    def __getitem__(self, index) -> torch.Tensor:
        idx = self.data_index[index]
        feat_idx = self.ldp.pair_index[idx]
        feature = self.ldp.get_feature(feat_idx, copy = True)
        feature = torch.from_numpy(feature).float()

        return feature

    def __iter__(self):
        self.batch_iter = gen_batches(self.num_samples, self.batch_size)
        
        return self
    
    def __next__(self):
        b_idx = next(self.batch_iter)
        data = self[b_idx]

        return b_idx, data
    
    def __len__(self):
        return self.num_samples
    


def get_device(idx: int = 0):
    if idx == -1:
        return torch.device("cpu")
    elif idx in range(torch.cuda.device_count()):
        return torch.device(idx)
    elif idx >= torch.cuda.device_count():
        raise RuntimeError("CUDA error: invalid device ordinal")
    else:
        raise ValueError("Invalid value for device index")



def cdist_rv(
    XA: Union[torch.Tensor, np.ndarray], 
    XB: Union[torch.Tensor, np.ndarray],
    p: float = 2.0,
    device: torch.device = torch.device("cuda")
) -> torch.Tensor:
    XA = torch.as_tensor(XA, dtype = torch.float32, device = device)
    XB = torch.as_tensor(XB, dtype = torch.float32, device = device)
    if len(XA.shape) == 1:
        XA = XA.reshape(1, -1)
    if len(XB.shape) == 1:
        XB = XB.reshape(1, -1)
    n = int(XA.shape[1]/2)
    XA_t = torch.cat((XA[:, n:], XA[:, :n]), 1)
    dist = torch.cdist(XA, XB, p)
    dist_t = torch.cdist(XA_t, XB, p)

    return torch.minimum(dist, dist_t)


# todo 和cupy的实现方法比较结果是否一致，函数的一些变量命名和代码可以再优化一下
def knn(
    ldp: Union[LabeledDataUnit, LabeledDataDoublet],
    k: int = 5,
    batch_size: int = 32,
    device: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    device = get_device(device)
    label = ldp.get_label(copy = True)
    pos_index = np.where(label)[0]
    neg_index = np.where(~label)[0]

    pos_loader = DataLoader(ldp, pos_index, batch_size = batch_size)
    neg_loader = DataLoader(ldp, neg_index, batch_size = 2**21)

    pos_feature = pos_loader[:]
    pos_dist = cdist_rv(pos_feature, pos_feature, device = device)
    threshold = torch.kthvalue(pos_dist, k = k + 1, keepdim = True)[0]

    row_index_total = []
    col_index_total = []
    for batch_r, pos_feat in pos_loader:
        for batch_c, neg_feat in neg_loader:
            dist = cdist_rv(pos_feat, neg_feat, device = device)
            row_index, col_index = torch.nonzero(torch.le(dist, threshold[batch_r]), as_tuple = True)
            row_index.add_(batch_r.start)
            col_index.add_(batch_c.start)
            row_index_total.extend(row_index.tolist())
            col_index_total.extend(col_index.tolist())
        
    marked_row_index = pos_index[row_index_total]
    marked_col_index = neg_index[col_index_total]
    BD_neg_index = np.unique(marked_col_index)

    return marked_row_index, marked_col_index, BD_neg_index


