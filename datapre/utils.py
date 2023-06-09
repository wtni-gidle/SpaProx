from typing import Optional, Tuple, Union

import scanpy as sc
from anndata import AnnData

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import random
import torch
from sklearn.model_selection import train_test_split


def setup_seed(seed: int = 38) -> None:
    """
    Globally set the seed for generating random numbers.
    
    Parameters
    ----------
    seed : int, default=38
        The desired seed.

    Returns
    -------
    `None`.
    """

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def feature_plot(adata: AnnData) -> None:
    """
    Draw a box plot to show `total_counts` and `n_genes_by_counts` for `adata`.

    Parameters
    ----------
    adata : AnnData
        The data for plotting.

    Returns
    -------
    `None`.
    """

    _, axs = plt.subplots(1, 2, figsize = (10, 5))
    plt.subplots_adjust(wspace = 0.5)
    if "total_counts" in adata.obs.keys():
        sns.boxplot(y = adata.obs["total_counts"], color = "green", width = 0.5, ax = axs[0])
    if "n_genes_by_counts" in adata.obs.keys():
        sns.boxplot(y = adata.obs["n_genes_by_counts"], color = "green", width = 0.5, ax = axs[1])



def filter(
    adata: AnnData, 
    min_counts: Optional[int] = None, 
    max_counts: Optional[int] = None, 
    min_genes: Optional[int] = None, 
    max_genes: Optional[int] = None, 
    min_cells: Optional[int] = None, 
    mt: Optional[float] = None
) -> AnnData:
    """
    Filter cell and gene outliers.
    - Keep cells with `total_counts` between `min_counts` and `max_counts`;
    - Keep cells with `n_genes_by_counts` between `min_genes` and `max_genes`;
    - Keep genes with `n_cells_by_counts` ≥ `min_cells`;
    - Keep cells with `pct_counts_mt` ≤ `mt`.

    Possible inputs for `min_counts`, `max_counts`, `min_genes` and `max_genes` are:
    - integer(>=0), the value itself;
    - `-1`, use `custom method` to determine values for filtering;
    - `None`, not to filter;

    `custom method` : Values outside :math:`(Q1 - 1.5*IQR, Q3 + 1.5*IQR)` are outliers.

    Parameters
    ----------
    adata : AnnData
        The data for filtering.

    min_counts : Optional[int], default=None
        Minimum number of counts required for a cell to pass filtering.

    max_counts : Optional[int], default=None
        Maximum number of counts required for a cell to pass filtering.

    min_genes : Optional[int], default=None
        Minimum number of genes expressed required for a cell to pass filtering.

    max_genes : Optional[int], default=None
        Maximum number of genes expressed required for a cell to pass filtering.

    min_cells : Optional[int], default=None
        Minimum number of cells expressed required for a gene to pass filtering.

    mt : Optional[int], default=None
        Maximum percentage of counts for mitochondrial gene required for a cell to pass filtering.

    Returns
    -------
    adata
        The filterd adata.
    """

    # region custom method
    q1 = adata.obs["total_counts"].quantile(0.25)
    q3 = adata.obs["total_counts"].quantile(0.75)
    if min_counts == -1:
        min_counts = int(
            max(
                q1 - 1.5 * (q3 - q1), 
                0
            )
        )

    if max_counts == -1:
        max_counts = int(
            min(
                q3 + 1.5 * (q3 - q1),
                max(adata.obs["total_counts"])
            )
        )
    
    q1 = adata.obs["n_genes_by_counts"].quantile(0.25)
    q3 = adata.obs["n_genes_by_counts"].quantile(0.75)

    if min_genes == -1:
        min_genes = int(
            max(
                q1 - 1.5 * (q3 - q1), 
                0
            )
        )

    if max_genes == -1:
        max_genes = int(
            min(
                q3 + 1.5 * (q3 - q1), 
                max(adata.obs["n_genes_by_counts"])
            )
        )
    # endregion

    # region filter
    if min_counts is not None:
        sc.pp.filter_cells(adata, min_counts = min_counts)

    if max_counts is not None:
        sc.pp.filter_cells(adata, max_counts = max_counts)
    
    if min_genes is not None:
        sc.pp.filter_cells(adata, min_genes = min_genes)

    if max_genes is not None:
        sc.pp.filter_cells(adata, max_genes = max_genes)

    if min_cells is not None:
        sc.pp.filter_genes(adata, min_cells = min_cells)

    if "pct_counts_mt" in adata.obs.keys() and mt is not None:
        adata = adata[adata.obs["pct_counts_mt"] <= mt].copy()
    # endregion
        
    return adata



def distance(adata: AnnData, pair_index) -> np.ndarray:
    """
    Calculate the distance between spots/cells.

    Parameters
    ----------
    adata : AnnData
        The data for processing.

    pair_index : np.ndarray
        2D `np.ndarray` of shape :math:`n_pairs` x 2, each row records the spot index of `spot pair` 
        whose distance is to be calculated.

    Returns
    -------
    distance
        1D `np.ndarray` of length `n_pairs`, record the distance.
    """

    location = adata.obs.loc[:, ["array_row", "array_col"]].values

    tmp = np.apply_along_axis(
        lambda x:np.abs(location[x[0]] - location[x[1]]), 
        1,
        pair_index
    )
    distance = np.apply_along_axis(
        lambda x:np.sqrt((x[0]**2 * 3 + x[1]**2)) / 2, 
        1, 
        tmp
    )

    return distance



def adata2seurat(
    adata: AnnData, 
    path: str = ""
) -> None:
    """
    Convert `Scanpy` to `Seurat`.

    Save gene expression matrix `mat` as an `h5` file and 
    save spatial location matrix `cord` as a `tsv` file.

    Parameters
    ----------
    adata : AnnData
        The data to convert.
        
    path : str, default=""
        The file save path.

    Returns
    -------
    `None`.
    """

    # region save mat
    mat_path = os.path.join(path, "count.h5")
    mat = pd.DataFrame(
        data = adata.X.todense(), 
        index = adata.obs_names, 
        columns = adata.var_names
    )
    mat.to_hdf(mat_path, "count")
    # endregion

    # region save cord
    cord_path = os.path.join(path, "spatial.tsv")
    cord = pd.DataFrame(
        data = adata.obsm["spatial"],
        index = adata.obs_names, 
        columns = ['x','y']
    )
    cord.to_csv(cord_path, sep = "\t")
    # endregion



def find_spa(path: str) -> list:
    """
    Extract `spagene` according to the save path.

    Parameters
    ----------
    path : str
        The save path of `spagene`.

    Returns
    -------
    spagene
        A list of spatially variable genes.
        
    """    

    with open(file = path, mode = "r") as f:
        spagene = f.readlines()

    spagene = [each.split("\n")[0] for each in spagene]
    
    return spagene



def preprocess(
    adata: AnnData, 
    gene: list, 
    normalize: bool = True, 
    log1p: bool = True,
    scale: bool = True
) -> AnnData:
    """
    Preprocess the `adata` and implement gene selection. The steps are as folows:
    - Normalize each spot/cell by total counts of 1e+04;
    - Compute :math:`X = \\log(X + 1)`;
    - Gene selection;
    - Scale data to unit variance and zero mean.

    Parameters
    ----------
    adata : AnnData
        The data to be processed.

    gene : list
        A list of genes for selection.

    normalize : bool, default=True
        Whether to normalize.

    log1p : bool, default=True
        Whether to Logarithmize.

    scale : bool, default=True
        Whether to scale.

    Returns
    -------
    adata
        The processed adata.
    """    

    # ! Using set to deduplicate will cause random order, and the result is different each time (setting seed is no use).
    # ! Use sort to fix the order in order to ensure consistent results.
    # * history code:
    # * if lrgene:
    # *     gene = list(lrgene | set(spagene))
    # *     gene.sort(key = spagene.index)
    # * else:
    # *     gene = spagene
    
    if normalize:
        sc.pp.normalize_total(adata, 10000)
    
    if log1p:
        sc.pp.log1p(adata)

    adata = adata[:, gene].copy()

    if scale:
        sc.pp.scale(adata)

    return adata



def split_dataset(
    label: np.ndarray, 
    feature: np.ndarray, 
    test_frac: float = 0.1, 
    valid_frac: Optional[float] = None
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
           Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Split the dataset in a stratified fashion.

    Parameters
    ----------
    label : np.ndarray
        The label of dataset.

    feature : np.ndarray
        The feature of dataset.

    test_frac : float, default=0.1
        The proportion of the test set.

    valid_frac : Optional[float], default=None
        The proportion of the validation set. :math:`valid_frac = Validation / (Train + Validation)`.

    Returns
    -------
    Depending on `valid_frac`, return four or six arrays as follows.
    - train_feature
    - train_label
    - val_feature (if `valid_frac` is given)
    - val_label (if `valid_frac` is given)
    - test_feature
    - test_label
    """

    X_train, X_test, y_train, y_test = train_test_split(
        feature, 
        label, 
        test_size = test_frac, 
        stratify = label
    )

    if valid_frac:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, 
            y_train, 
            test_size = valid_frac, 
            stratify = y_train
        )
        return X_train, y_train, X_val, y_val, X_test, y_test

    return X_train, y_train, X_test, y_test

 

def mirror_copy(feature):
    n = int(feature.shape[1]/2)
    return np.concatenate((feature[:, n:], feature[:, :n]), 1)