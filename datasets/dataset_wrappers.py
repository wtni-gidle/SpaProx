from typing import Tuple, Union
import torch
from torch.utils.data import Dataset
from .data_process import LabeledDataUnit, LabeledDataDoublet, UnlabeledDataUnit, UnlabeledDataDoublet

# todo 可以尝试一下多切片的训练，可以参考celery
class LabeledDataset(Dataset):
    def __init__(self, ldp: Union[LabeledDataUnit, LabeledDataDoublet]) -> None:
        self.ldp = ldp
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        label_idx = self.ldp.data_index[index]
        feat_idx = self.ldp.pair_index[index]
        label = self.ldp.get_label(label_idx, copy = True)
        feature = self.ldp.get_feature(feat_idx, copy = True)

        label = torch.from_numpy(label).long()
        feature = torch.from_numpy(feature).float()

        return feature, label
    
    def __len__(self):
        return len(self.ldp.data_index)
    


class UnlabeledDataset(Dataset):
    def __init__(self, udp: Union[UnlabeledDataUnit, UnlabeledDataDoublet]) -> None:
        self.udp = udp

    def __getitem__(self, index) -> torch.Tensor:
        feat_idx= self.udp.pair_index[index]
        feature = self.udp.get_feature(feat_idx, copy = True)

        feature = torch.from_numpy(feature).float()

    def __len__(self):
        return len(self.udp.data_index)
    
