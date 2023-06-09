from typing import Tuple, Optional
import numpy as np
from itertools import combinations, product
from sklearn.model_selection import train_test_split
from datapre import utils
from anndata import AnnData



class BaseData:
    """
    A data process unit.

    A data process unit is a simple module for building a dataset. 
    Given `adata` and `spot_index`, it can generate spot pairs, calculate the distance, 
    and resample. It performs independent preprocessing on the `adata` subset, and obtain features and Label.

    Attributes
    ----------
    adata : AnnData
        The original data.

    spot_index : Optional[np.ndarray], default=None
        The spot subset index of `adata`. If None, `spot_index` is the total spot index of `adata`.
    
    adata_son : AnnData
        The subset of the data.

    pair_index_total : np.ndarray
        2D `np.ndarray` of shape :math:`n_pairs` x 2, each row records the spot index of `spot pair`
        (based on `adata`). :math:`n_pairs = n_spots x (n_spots - 1) / 2`.
    
    pair_index_total_son : np.ndarray
        2D `np.ndarray` of shape :math:`n_pairs` x 2, each row records the spot index of `spot pair`
        (based on `adata_son`).

    distance : np.ndarray
        1D `np.ndarray` of length `n_pairs`, record the distance.

    label_total : np.ndarray
        1D `np.ndarray` of length `n_pairs`, record the label.

    pair_index_pos : np.ndarray
        2D `np.ndarray`, each row records the spot index of the positive `spot pair`
        (based on `adata`).

    pair_index_pos_son : np.ndarray
        2D `np.ndarray`, each row records the spot index of the positive `spot pair`
        (based on `adata_son`).
    
    pos_index : np.ndarray
        1D `np.ndarray`, record the positive index.
    
    pair_index_neg : np.ndarray
        2D `np.ndarray`, each row records the spot index of the (sampled) negative `spot pair`
        (based on `adata`).
    
    pair_index_total_son : np.ndarray
        2D `np.ndarray`, each row records the spot index of the (sampled) negative `spot pair`
        (based on `adata_son`).
    
    neg_index : np.ndarray
        1D `np.ndarray`, record the negative index.
    
    pair_index : np.ndarray
        2D `np.ndarray`, each row records the spot index of the selected `spot pair`
        (based on `adata`).
    
    pair_index_son : np.ndarray
        2D `np.ndarray`, each row records the spot index of the selected `spot pair`
        (based on `adata_son`).

    sample_index : np.ndarray
        1D `np.ndarray`, record the selected index.
    """

    def __init__(
        self, 
        adata: AnnData, 
        neighbor_dis: float, 
        spot_index: Optional[np.ndarray] = None
    ) -> None:
        self.adata = adata
        self.neighbor_dis = neighbor_dis

        if spot_index is None:
            spot_index = np.arange(len(adata))
        
        self.spot_index = spot_index

        self.adata_son = self.adata[self.spot_index].copy()
        self.spot2pair()

    def spot2pair(self):
        """
        Generate `pair_index_total` (and `pair_index_total_son`) based on `spot_index`.
        """
        self.pair_index_total = None
        self.pair_index_total_son = None
        self.total_index = np.arange(len(self.pair_index_total))

        raise NotImplementedError('spot2pair needs to be redefined')
    
    def calc_distance(self):
        """
        Calculate the distance according to `pair_index_total`.
        """
        self.distance = utils.distance(self.adata, pair_index = self.pair_index_total)

        self.label_total = self.distance <= self.neighbor_dis

    def get_pos(self):
        """
        Get `pair_index_pos`.
        """
        self.pos_index = np.compress(self.label_total, self.total_index, 0)
        self.pair_index_pos = np.take(self.pair_index_total, self.pos_index, 0)
        if hasattr(self, "pair_index_total_son"):
            self.pair_index_pos_son = np.take(self.pair_index_total_son, self.pos_index, 0)

    def get_neg(self, neg_size: Optional[float] = None):
        """
        Get `pair_index_neg`.

        Parameters
        ----------
        neg_size : Optional[float], default=None
            If `None`, select the total negative index; if given, downsample the negative samples according to
            the ratio of :math:`pos : neg = 1 : neg_size`.
        """
        pos_num = sum(self.label_total)

        if neg_size:
            self.neg_index = np.random.choice(
                self.total_index[~self.label_total], 
                int(neg_size * pos_num), 
                replace = False
            )
        else:
            self.neg_index = np.compress(~self.label_total, self.total_index, 0)

        self.pair_index_neg = self.pair_index_total[self.neg_index]
        if hasattr(self, "pair_index_total_son"):
            self.pair_index_neg_son = np.take(self.pair_index_total_son, self.neg_index, 0)

    def get_data(self, neg_size: Optional[float] = None, mirror: bool = True):
        """
        Get `pair_index`.

        Parameters
        ----------
        neg_size : Optional[float], default=None
            If `None`, select the total negative index; if given, downsample the negative samples according to
            the ratio of :math:`pos : neg = 1 : neg_size`.

        mirror : bool, default=True
            Whether to "mirror copy" positive samples.
        """
        self.get_pos()
        self.get_neg(neg_size = neg_size)
        self.data_index = np.concatenate((self.pos_index, self.neg_index))
        # self.pair_index = self.pair_index_total[self.data_index]
        self.pair_index = np.take(self.pair_index_total, self.data_index, 0)
        if hasattr(self, "pair_index_total_son"):
            # self.pair_index_son = self.pair_index_total_son[self.data_index]
            self.pair_index_son = np.take(self.pair_index_total_son, self.data_index, 0)

        if mirror:
            self.mirror_copy()

    def mirror_copy(self):
        """
        在原来的基础上添加正样本
        """
        self.data_index = np.concatenate((self.data_index, self.pos_index))
        self.pair_index = np.concatenate((self.pair_index, np.flip(self.pair_index_pos, axis = 1)))
        if hasattr(self, "pair_index_son"):
            self.pair_index_son = np.concatenate((self.pair_index_son, np.flip(self.pair_index_pos_son, axis = 1)))
    
    def pop(self, obj):
        self.data_index = np.delete(self.data_index, obj, 0)
        self.pair_index = np.delete(self.pair_index, obj, 0)
        if hasattr(self, "pair_index_son"):
            self.pair_index_son = np.delete(self.pair_index_son, obj, 0)
        
    def update(self, obj):
        self.data_index = np.take(self.data_index, obj, 0)
        self.pair_index = np.take(self.data_index, obj, 0)
        if hasattr(self, "pair_index_son"):
            self.pair_index_son = np.take(self.pair_index_son, obj)

    
    def get_label(self, index: Optional[np.ndarray] = None, copy: bool = False):
        """
        Get `label`.

        Parameters
        ----------
        index : Optional[np.ndarray], default=None
            The desired index that needs to return label. If `None`, `index` = `self.sample_index`.

        copy : bool, default=False
            Whether to return `label`.

        Returns
        -------
        Return `label`, depending on `inplace`.
        """

        if index is None:
            index = self.data_index

        label = np.take(self.label_total, index, 0)
        if copy:
            return label
        else:
            self.label = label
 
    def get_feature(self, index: Optional[np.ndarray] = None, copy: bool = False):
        """
        Get `feature`.

        Parameters
        ----------
        index : Optional[np.ndarray], default=None
            The desired index that needs to return feature. If `None`, `index` = `self.sample_index`.

        copy : bool, default=False
            Whether to return `feature`.

        Returns
        -------
        Return `feature`, depending on `inplace`.
        """

        if index is None:
            index = self.pair_index_son

        feature = np.apply_along_axis(
            lambda x:np.concatenate((self.count[x[0]], self.count[x[1]])), 
            1,
            index
        )
        if copy:
            return feature
        else:
            self.feature = feature

    def preprocess(
        self, 
        gene: list, 
        normalize: bool = True, 
        log1p: bool = True, 
        scale: bool = True
    ):
        """
        See :ref:`utils.preprocess`.

        Parameters
        ----------
        gene : list
            A list of genes for selection.

        normalize : bool, default=True
            Whether to normalize.

        log1p : bool, default=True
            Whether to Logarithmize.

        scale : bool, default=True
            Whether to scale.
        """

        self.adata_son = utils.preprocess(
            self.adata_son, 
            gene = gene,
            normalize = normalize,
            log1p = log1p,
            scale = scale
        )
        self.count = self.adata_son.to_df().values



class DataUnit(BaseData):

    def spot2pair(self):
        """
        Generate `pair_index_total` (and `pair_index_total_son`) based on `spot_index`.
        """
        self.pair_index_total = np.array(list(
            combinations(
                self.spot_index, 2
            )
        ))
        self.pair_index_total_son = np.array(list(
            combinations(
                range(len(self.spot_index)), 2
            )
        ))
        self.total_index = np.arange(len(self.pair_index_total))



class DataDoublet(BaseData):

    def __init__(
        self,
        adata: AnnData, 
        neighbor_dis: float, 
        unit1: DataUnit, 
        unit2: DataUnit
    ) -> None:
        self.adata = adata
        self.neighbor_dis = neighbor_dis
        self.spot_index = [unit1.spot_index, unit2.spot_index]
        self.update_count(unit1, unit2)
        self.spot2pair()



    def update_count(
        self, 
        unit1: DataUnit, 
        unit2: DataUnit
    ) -> None:
        shape = (self.adata.shape[0], unit1.count.shape[1])
        count = np.empty(shape)
        count[unit1.spot_index] = unit1.count.copy()
        count[unit2.spot_index] = unit2.count.copy()
        self.count = count



    def spot2pair(self):
        self.pair_index_total = np.array(list(
            product(
                self.spot_index[0], self.spot_index[1]
            )
        ))
        self.total_index = np.arange(len(self.pair_index_total))
    
        

    def get_feature(self, index =  None, copy = False):
        if index is None:
            index = self.pair_index
        if copy:
            return super().get_feature(index = index, copy = copy)
        else:
            super().get_feature(index = index, copy = copy)








def dataset_blind(
    adata, 
    gene, 
    neighbor_dis,
    neg_size = 10,
    mirror = False
):
    train_spot_index, test_spot_index = train_test_split(np.arange(adata.n_obs), test_size=0.2)
    train_data = DataUnit(adata, neighbor_dis=neighbor_dis, spot_index=train_spot_index)
    test_data = DataUnit(adata, neighbor_dis=neighbor_dis, spot_index=test_spot_index)

    train_data.calc_distance()
    train_data.get_data(neg_size=neg_size, mirror=mirror)
    train_data.preprocess(gene=gene,normalize=True, log1p=True,scale=True)
    train_data.get_label()
    train_data.get_feature()

    test_data.calc_distance()
    test_data.get_data(neg_size=None, mirror=False)
    test_data.preprocess(gene=gene,normalize=True, log1p=True,scale=True)
    test_data.get_label()
    test_data.get_feature()

    tt_data = DataDoublet(adata, neighbor_dis=neighbor_dis, unit1=train_data, unit2=test_data)
    tt_data.calc_distance()
    tt_data.get_data(neg_size=None, mirror=False)
    tt_data.get_label()
    tt_data.get_feature()

    return train_data, test_data, tt_data


def dataset(
    adata: AnnData, 
    gene: list, 
    neighbor_dis: float,
    neg_size: Optional[float] = 2,
    mirror: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate the dataset as follows:
    - Calculate the distance;
    - Log-normalize and scale the data;
    - Downsample the negative samples;
    - Split the dataset.
    - "mirror copy" the positive samples of the training set.

    Parameters
    ----------
    adata : AnnData
        The data ro make dataset.

    gene : list
        A list of spatially variable genes.

    neighbor_dis : float
        Cell neighborhood radius.

    neg_size : Optional[float], default=2
        If `None`, select the total negative index; if given, downsample the negative samples according to
        the ratio of :math:`pos : neg = 1 : neg_size`.

    mirror : bool, default=True
        Whether to "mirror copy" positive samples in training set.

    Returns
    -------
    Return the following arrays.
    - train_feature
    - train_label
    - test_feature, the first 2 columns record the `pair_index`.
    - test_label
    - distance, the total distance.
    """
    
    data = DataUnit(adata, neighbor_dis = neighbor_dis)
    data.calc_distance()
    distance = data.distance
    data.get_pdata(neg_size = neg_size, mirror = False)
    data.preprocess(gene = gene, normalize = True, log1p = True, scale = True)
    label = data.get_label(copy = True)
    index = np.concatenate((data.pair_index, data.pair_index_son), axis = 1)
    train_index, train_label, test_index, test_label = utils.split_dataset(label, index)

    test_feature = data.get_feature(test_index[:, -2:], copy = True)
    test_feature = np.concatenate((test_index[:, :2], test_feature), axis = 1)

    train_index = train_index[:, -2:].copy()

    if mirror:
        mirror_index = np.flip(train_index[train_label], axis = 1)
        train_index = np.concatenate((train_index, mirror_index))
        train_label = np.concatenate((train_label, train_label[train_label]))

    train_feature = data.get_feature(train_index, copy = True)
    

    return train_feature, train_label, test_feature, test_label, distance



