from typing import Union, Tuple
import torch
import numpy as np
from sklearn.utils import gen_batches
from datasets.data_process import LabeledDataUnit, LabeledDataDoublet
from copy import deepcopy
from gpu_mem_track import MemTracker
gpu_tracker = MemTracker() 

class DataLoader():
    def __init__(
        self, 
        ldp: Union[LabeledDataUnit, LabeledDataDoublet], 
        data_index: list,
        batch_size: int = 32
    ) -> None:
        self.ldp = ldp
        self.data_index = data_index
        self.batch_size = batch_size
        self.num_samples = len(self.data_index)
    
    def __getitem__(self, index):
        idx = self.data_index[index]
        feat_idx = self.ldp.pair_index[idx]
        feature = self.ldp.get_feature(feat_idx, copy = True)

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
    XA: torch.Tensor, 
    XB: torch.Tensor,
    p: float = 2.0
) -> torch.Tensor:
    if len(XA.shape) == 1:
        XA = XA.reshape(1, -1)
    if len(XB.shape) == 1:
        XB = XB.reshape(1, -1)
    n = int(XA.shape[1]/2)
    XA_t = torch.cat((XA[:, n:], XA[:, :n]), 1)
    dist = torch.cdist(XA, XB, p)
    dist_t = torch.cdist(XA_t, XB, p)

    return torch.minimum(dist, dist_t)



def knn(
    ldp: Union[LabeledDataUnit, LabeledDataDoublet],
    k: int = 5,
    batch_size: int = 64,
    device: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    gpu_tracker.track() 
    device = get_device(device)
    label = ldp.get_label(copy = True)
    pos_index = np.where(label)[0].tolist()
    neg_index = np.where(~label)[0].tolist()

    ldp = to_device(ldp, device)
    pos_loader = DataLoader(ldp, pos_index, batch_size = batch_size)
    neg_loader = DataLoader(ldp, neg_index, batch_size = 2**21)

    pos_feature = pos_loader[:]
    pos_dist = cdist_rv(pos_feature, pos_feature)
    threshold = torch.kthvalue(pos_dist, k = k + 1, keepdim = True)[0]

    row_index_total = []
    col_index_total = []
    for batch_r, pos_feat in pos_loader:
        for batch_c, neg_feat in neg_loader:
            gpu_tracker.track() 
            dist = cdist_rv(pos_feat, neg_feat)
            row_index, col_index = torch.nonzero(torch.le(dist, threshold[batch_r]), as_tuple = True)
            row_index.add_(batch_r.start)
            col_index.add_(batch_c.start)
            row_index_total.extend(row_index.tolist())
            col_index_total.extend(col_index.tolist())
            torch.cuda.empty_cache()
    marked_row_index = np.array(pos_index)[row_index_total]
    marked_col_index = np.array(neg_index)[col_index_total]
    BD_neg_index = np.unique(marked_col_index)

    return marked_row_index, marked_col_index, BD_neg_index

#将实例转移到gpu上，这样取feature就不用多次转移，可以直接计算
def to_device(obj, device):
    obj = deepcopy(obj)

    custom_attr = [a for a in dir(obj) if not a.startswith('__')]
    attributes = [attr for attr in custom_attr if not callable(getattr(obj, attr))]
    for attr in attributes:
        value = getattr(obj, attr)
        if isinstance(value, np.ndarray):
            if value.dtype in (np.int32, np.int64):
                setattr(obj, attr, torch.as_tensor(value, dtype = torch.long, device = device))
            else:
                setattr(obj, attr, torch.as_tensor(value, device = device))
    
    return obj