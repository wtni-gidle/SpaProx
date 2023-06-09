import numpy as np
from cupyx.scipy.spatial.distance import cdist
import cupy as cp


# def metric(u: np.ndarray, v: np.ndarray, **kwargs):
#     u = u.reshape(1, -1)
#     n = int(u.shape[1]/2)
#     u_t = np.concatenate((u[n:], u[:n]), 1)
#     v = v.reshape(1, -1)
#     d1 = cdist(u, v, **kwargs)
#     d2 = cdist(u_t, v, **kwargs)
    
#     return min(d1, d2)

# def metric(u, v):
#     n = int(u.shape[0]/2)
#     u_t = np.concatenate((u[n:], u[:n]))
#     d1 = np.linalg.norm(u - v)
#     d2 = np.linalg.norm(u_t - v)

#     return min(d1, d2)

# def metric_np(u, v):

#     return np.linalg.norm(u - v)

# cv = RepeatedStratifiedKFold(n_splits = 10, n_repeats = 3)
# scores = cross_val_score(model, X, y, scoring='roc_auc', cv = cv, n_jobs = -1)

def cdist_rv(XA, XB, metric = "euclidean", **kwargs):
    XA = cp.asarray(XA)
    XB = cp.asarray(XB)
    if len(XA.shape) == 1:
        XA = XA.reshape(1, -1)
    if len(XB.shape) == 1:
        XB = XB.reshape(1, -1)
    n = int(XA.shape[1]/2)
    XA_t = cp.concatenate((XA[:, n:], XA[:, :n]), 1)
    dist = cdist(XA, XB, metric, **kwargs)
    dist_t = cdist(XA_t, XB, metric, **kwargs)

    return cp.minimum(dist, dist_t)



def eliminate_BD_neg(feature, label, k = 1, batch_size = 128):
    feature = cp.asarray(feature)
    label = cp.asarray(label)

    pos_feature = cp.compress(label, feature, axis = 0)
    neg_feature = cp.compress(~label, feature, axis = 0)
    n = pos_feature.shape[0]
    pos_dist = cdist_rv(pos_feature, pos_feature)

    threshold = cp.sort(pos_dist, axis = 1)[:, [k]]

    marked_row_index = cp.array([], dtype = "int64")
    marked_col_index = cp.array([], dtype = "int64")
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        dist = cdist_rv(pos_feature[s: e], neg_feature)
        row_index, col_index = cp.nonzero(cp.less_equal(dist, threshold[s: e]))

        cp.add(row_index, s, out = row_index)
        marked_row_index = cp.concatenate((marked_row_index, row_index))
        marked_col_index = cp.concatenate((marked_col_index, col_index))

    marked_col_index = cp.nonzero(~label)[0][marked_col_index]
    marked_neg_index = cp.unique(marked_col_index)

    return marked_row_index, marked_col_index, marked_neg_index


