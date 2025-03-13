from pathlib import Path

import cv2
import torch
import torchvision.transforms.v2 as T

from torch.utils.data import Dataset


class HistologyDataset(Dataset):
    def __init__(self, images: list[Path]) -> None:
        assert all(p.exists() for p in images)

        self.images = images
        self.transforms = T.Compose([
            T.ToImage(), 
            T.ToDtype(torch.float32, scale=True),
            T.Resize((256, 256)),
            T.RandomHorizontalFlip()
        ])

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = cv2.imread(str(self.images[index]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.transforms(image)
