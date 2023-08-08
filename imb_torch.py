from typing import Union
import torch
import numpy as np
from sklearn.utils import gen_batches
from math import ceil


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
    device: int = 0
) -> torch.Tensor:
    XA = torch.as_tensor(XA, device = device)
    XB = torch.as_tensor(XB, device = device)
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
def eliminate_BD_neg(
    feature: np.ndarray, 
    label: np.ndarray, 
    k: int = 1,
    batch_size: int = 64,
    device: int = 0
):
    device = get_device(device)
    pos_index = np.where(label)[0]
    neg_index = np.where(~label)[0]
    pos_feature = np.take(feature, pos_index, 0)
    pos_dist = cdist_rv(pos_feature, pos_feature, device = device)
    threshold = torch.kthvalue(pos_dist, k = k + 1, keepdim = True)[0]

    col_batch = gen_batches(len(neg_index), ceil(len(neg_index)/2))
    row_batch = gen_batches(len(pos_index), batch_size)

    row_index_total = []
    col_index_total = []
    for batch_r in row_batch:
        pos_feature = feature[pos_index[batch_r]]
        for batch_c in col_batch:
            neg_feature = feature[neg_index[batch_c]]
            dist = cdist_rv(pos_feature, neg_feature, device = device)
            row_index, col_index = torch.nonzero(torch.le(dist, threshold[batch_r]), as_tuple=True)
            row_index.add_(batch_r.start)
            col_index.add_(batch_c.start)
            row_index_total.extend(row_index.tolist())
            col_index_total.extend(col_index.tolist())
        
    marked_row_index = pos_index[row_index_total]
    marked_col_index = neg_index[col_index_total]
    BD_neg_index = np.unique(marked_col_index)

    return marked_row_index, marked_col_index, BD_neg_index


