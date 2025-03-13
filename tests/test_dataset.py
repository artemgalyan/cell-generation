from pathlib import Path

import torch

from src import HistologyDataset


def test_dataset():
    dataset = HistologyDataset([
        Path('data/train/epi/Normal_001_tilec_EPI_4831.png')
    ])
    item = dataset[0]
    assert item.shape == (3, 256, 256)
    assert isinstance(item, torch.Tensor)
    assert torch.all((0 <= item) & (item <= 1))