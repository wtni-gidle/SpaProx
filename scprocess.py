from typing import Optional, Tuple
from sklearn.base import BaseEstimator
from anndata import AnnData

import itertools
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform


def CreatePair(
    adata_sc: AnnData, 
    sample_num: int = 2000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate cell pairs between different cell types. Here use a method to sample cell pairs.
    
    For Cell Type A with :math:`n_A` cells and Cell Type B with :math:`n_B` cells:
    - If :math:`n_A x n_B < sample_num`, then select all cell pairs;
    - If :math:`n_A x n_B >= sample_num`, then sample 2000 cell pairs. 

    Parameters
    ----------
    adata_sc : AnnData
        The single cell data.

    sample_num : int, default=2000
        Maximum number of cell pairs between different cell types.
        
    Returns
    -------
    Return the following arrays.

    feature_index
        2D `np.ndarray` of shape `n_pairs` x 2, record the cell index.

    pair_cluster
        2D `np.ndarray` of shape `n_pairs` x 2, dtype=`object`. Record the cell cluster.
    """

    cluster = adata_sc.obs["clusters"].unique()
    cluster_counts = adata_sc.obs["clusters"].value_counts()
    feature_index = []
    pair_cluster = []

    for each in list(itertools.combinations(cluster, 2)):
        counts_1 = cluster_counts[each[0]]
        counts_2 = cluster_counts[each[1]]
        num = sample_num
        if counts_1 * counts_2 < num:
            num = counts_1 * counts_2
        index_1 = np.where(adata_sc.obs["clusters"] == each[0])[0]
        index_2 = np.where(adata_sc.obs["clusters"] == each[1])[0]
        sample_index = random.sample(list(itertools.product(index_1, index_2)), num)
        feature_index.extend(sample_index)
        pair_cluster.extend([each] * num)

    feature_index = np.array(feature_index, dtype = np.int32)
    pair_cluster = np.array(pair_cluster, dtype = object)

    return feature_index, pair_cluster



def batch_predict(
    feature_index: np.ndarray, 
    count: np.ndarray, 
    model: BaseEstimator, 
    num_features: Optional[int] = None, 
    batch_size: int = 1024
):
    """
    Batch Predictions Using the trained model.

    Parameters
    ----------
    feature_index : np.ndarray
        2D `np.ndarray` of shape `n_pairs` x 2, record the cell index.

    count : np.ndarray
        The gene expression matrix of adata.

    model : BaseEstimator
        The trained estimator.

    num_features : Optional[int], default=None
        `num_features` of `model`.

    batch_size : int, default=1024
        Number of samples for each training.

    Returns
    -------
    pred_result
        A list records the prediction result.
    """

    n_samples = feature_index.shape[0]
    pred_result = []
    for i in range(0, n_samples, batch_size):
        batch_index = feature_index[i: min(i + batch_size, n_samples)]
        feature = np.apply_along_axis(
            lambda x:np.concatenate((count[x[0]], count[x[1]])), 
            1, 
            batch_index
        )
        if num_features:
            start = int(feature.shape[1] / 2)
            feature = np.concatenate(
                (feature[:, :num_features], feature[:, start:(start + num_features)]),
                1
            )
        pred = model.predict(feature)
        pred_result.extend(pred)

    return pred_result



def cluster_distance(
    adata_sc: AnnData, 
    pred_result: list, 
    pair_cluster: np.ndarray
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Calculate `Cell Proximity Matrix` according to the prediction result.

    Parameters
    ----------
    adata_sc : AnnData
        The single cell data.

    pred_result : list
        A list records the prediction result.
        
    pair_cluster : np.ndarray
        2D `np.ndarray` of shape `n_pairs` x 2, dtype=`object`. Record the cell cluster.

    Returns
    -------
    Return the following dataframes.

    positive_mat
        A `DataFrame` record the number of positive pairs.
    total_mat
        A `DataFrame` record the number of total pairs.
    mean_mat
        A `DataFrame` record the mean of positive pairs (`Proximity`).
    """    

    cluster_between_index = pd.MultiIndex.from_arrays(pair_cluster.T, names = ["first", "second"])
    cluster_between_pred = pd.Series(pred_result, cluster_between_index)
    cluster_between_pred = cluster_between_pred.groupby(["first", "second"])

    cluster_between_positive = cluster_between_pred.agg(np.sum)
    cluster_between_total = cluster_between_pred.agg(len)
    cluster_between_mean = cluster_between_pred.agg(np.mean)

    positive_mat = total_mat = mean_mat = pd.DataFrame(
        index = adata_sc.obs["clusters"].unique(),
        columns = adata_sc.obs["clusters"].unique(),
        dtype = float
    )

    for each in cluster_between_positive.index:
        positive_mat.loc[each] = cluster_between_positive.loc[each]
        positive_mat.loc[each[1], each[0]] = positive_mat.loc[each]

    for each in cluster_between_total.index:
        total_mat.loc[each] = cluster_between_total.loc[each]
        total_mat.loc[each[1], each[0]] = total_mat.loc[each]

    for each in cluster_between_mean.index:
        mean_mat.loc[each] = cluster_between_mean.loc[each]
        mean_mat.loc[each[1], each[0]] = mean_mat.loc[each]

    np.fill_diagonal(positive_mat.values, 1.0)
    np.fill_diagonal(total_mat.values, 1.0)
    np.fill_diagonal(mean_mat.values, 1.0)
    mean_mat = mean_mat.round(4)
    
    return positive_mat, total_mat, mean_mat



def cluster(
    close_mat: pd.DataFrame, 
    orientation: str = "right",
    dirname: Optional[str] = None
) -> None:
    """
    Cluster and plot dendrogram.

    Parameters
    ----------
    close_mat : pd.DataFrame
        Cell Proximity Matrix.

    orientation : str
        The direction to plot the dendrogram, which can be any of the following strings:
        - "top", plot the root at the top and plot descendent links going downwards;
        - "bottom", plot the root at the bottom and plot descendent links going upwards.
        - "left", plot the root at the left and plot descendent links going right.
        - "right", plot the root at the right and plot descendent links going left.

    dirname : Optional[str], default=None
        The directory name used to join save path if `dirname` is given.

    Returns
    -------
    `None`.
    """

    Z = linkage(squareform(1.0 - close_mat), "average")
    plt.figure(figsize = (5, 20))
    dendrogram(Z, orientation = orientation, labels = close_mat.index)
    if dirname:
        plt.savefig(dirname + "/dendrogram.svg", bbox_inches = 'tight')



def clustermap(
    close_mat: pd.DataFrame, 
    figsize: tuple,
    dirname: Optional[str] = None
):
    """
    Cluster and plot hierarchically-clustered heatmap.

    Parameters
    ----------
    close_mat : pd.DataFrame
        Cell Proximity Matrix.

    figsize : tuple
        Figure size.
    
    dirname : Optional[str], default=None
        The directory name used to join save path if `dirname` is given.

    Returns
    -------
    `None`.
    """

    Z = linkage(squareform(1.0 - close_mat), "average")
    ax = sns.clustermap(close_mat, row_linkage = Z, col_linkage = Z, xticklabels=1, yticklabels = 1,
                        cbar_pos=(0.9, 0.43, 0.03, 0.13), cmap = "YlOrRd",
                        tree_kws = {'linewidths':1.5})
    ax.figure.set_size_inches(*figsize)
    ax.ax_cbar.axes.set_title("Proximity", fontdict = {"fontsize":16})
    if dirname:
        plt.savefig(dirname + "/clustermap.svg", bbox_inches = 'tight')
    plt.show()


