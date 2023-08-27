from typing import Tuple, Union
import torch
from torch.utils.data import Dataset
from .data_process import LabeledDataUnit, LabeledDataDoublet, UnlabeledDataUnit, UnlabeledDataDoublet
import numpy as np


# *可以使用以下方式加载多个切片
# *ConcatDataset([ds1, ds2])

class LabeledDataset(Dataset):
    """
    Labeled dataset as input of DataLoader.
    It is initialized using LabeledDataUnit or LabeledDataDoublet.
    """
    def __init__(self, ldp: Union[LabeledDataUnit, LabeledDataDoublet]) -> None:
        self.ldp = ldp
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        label_idx = self.ldp.data_index[index]
        feat_idx = self.ldp.pair_index[index]
        label = self.ldp.get_label(label_idx, copy = True).long()
        feature = self.ldp.get_feature(feat_idx, copy = True)

        if isinstance(label, np.ndarray):
            label = torch.from_numpy(label).long()
            feature = torch.from_numpy(feature).float()

        feature = feature.reshape(-1, feature.shape[-1])

        return feature, label
    
    def __len__(self):
        return len(self.ldp.data_index)
    


class UnlabeledDataset(Dataset):
    """
    Unlabeled dataset as input of DataLoader.
    It is initialized using UnlabeledDataUnit or UnlabeledDataDoublet.
    """
    def __init__(self, udp: Union[UnlabeledDataUnit, UnlabeledDataDoublet]) -> None:
        self.udp = udp

    def __getitem__(self, index) -> torch.Tensor:
        feat_idx= self.udp.pair_index[index]
        feature = self.udp.get_feature(feat_idx, copy = True)
        
        if isinstance(feature, np.ndarray):
            feature = torch.from_numpy(feature).float()
        
        return feature

    def __len__(self):
        return len(self.udp.data_index)
    
