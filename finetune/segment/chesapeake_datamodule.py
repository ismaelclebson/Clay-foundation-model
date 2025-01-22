"""
DataModule for the Chesapeake Bay dataset for segmentation tasks.

This implementation provides a structured way to handle the data loading and
preprocessing required for training and validating a segmentation model.

Dataset citation:
Robinson C, Hou L, Malkin K, Soobitsky R, Czawlytko J, Dilkina B, Jojic N.
Large Scale High-Resolution Land Cover Mapping with Multi-Resolution Data.
Proceedings of the 2019 Conference on Computer Vision and Pattern Recognition
(CVPR 2019).

Dataset URL: https://lila.science/datasets/chesapeakelandcover
"""

import re
from pathlib import Path

import lightning as L
import numpy as np
import torch
import yaml
from box import Box
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

class ChesapeakeDataset(Dataset):
    def __init__(self, chip_dir, label_dir, metadata, platform, is_eval=False):
        self.chip_dir = Path(chip_dir)
        self.label_dir = Path(label_dir) if label_dir else None
        self.metadata = metadata
        self.is_eval = is_eval
        self.transform = self.create_transforms(
            mean=list(metadata[platform].bands.mean.values()),
            std=list(metadata[platform].bands.std.values()),
        )

        self.chips = [chip_path.name for chip_path in self.chip_dir.glob("*.npy")]
        if not is_eval:
            self.labels = [chip for chip in self.chips]

    def create_transforms(self, mean, std):
        return v2.Compose([
            v2.Normalize(mean=mean, std=std),
        ])

    def __len__(self):
        return len(self.chips)

    def __getitem__(self, idx):
        chip_name = self.chip_dir / self.chips[idx]
        chip = np.load(chip_name).astype(np.float32)

        sample = {
            "pixels": self.transform(torch.from_numpy(chip)),
            "time": torch.zeros(4),
            "latlon": torch.zeros(4),
        }

        if not self.is_eval:
            label_name = self.label_dir / self.labels[idx]
            label = np.load(label_name).astype(np.int64)
            #label_mapping = {0: 0, 3: 1, 4: 2, 9: 3, 11: 4, 12: 5, 19: 6, 21: 7, 25: 8, 29: 9, 33: 10} # mapbionas multclass-goias
            label_mapping = {0: 0, 1: 1}
            remapped_label = np.vectorize(label_mapping.get)(label)
            sample["label"] = torch.from_numpy(remapped_label[0])

        return sample

class ChesapeakeDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_chip_dir,
        train_label_dir,
        val_chip_dir,
        val_label_dir,
        metadata_path,
        batch_size,
        num_workers,
        platform,
        eval_chip_dir=None,
    ):
        super().__init__()
        self.train_chip_dir = train_chip_dir
        self.train_label_dir = train_label_dir
        self.val_chip_dir = val_chip_dir
        self.val_label_dir = val_label_dir
        self.eval_chip_dir = eval_chip_dir
        self.metadata = Box(yaml.safe_load(open(metadata_path)))
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.platform = platform

    def setup(self, stage=None):
        if stage in {"fit", None}:
            self.trn_ds = ChesapeakeDataset(
                self.train_chip_dir,
                self.train_label_dir,
                self.metadata,
                self.platform,
            )
            self.val_ds = ChesapeakeDataset(
                self.val_chip_dir,
                self.val_label_dir,
                self.metadata,
                self.platform,
            )
        if stage == "evaluate":
            self.eval_ds = ChesapeakeDataset(
                self.eval_chip_dir,
                None,
                self.metadata,
                self.platform,
                is_eval=True,
            )

    def train_dataloader(self):
        return DataLoader(
            self.trn_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def eval_dataloader(self):
        return DataLoader(
            self.eval_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
