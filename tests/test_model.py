import pytest

import numpy as np
import torch

from torch.optim import Adam

from src import UNet, sample


@pytest.mark.parametrize('batch_size', [1, 2, 4, 8])
@pytest.mark.parametrize('image_size', [256])
def test_model(batch_size: int, image_size: int) -> None:
    model = UNet(3).eval().cuda()
    t = torch.rand(batch_size, 1).cuda()
    image_batch = torch.randn(batch_size, 3, image_size, image_size, dtype=torch.float32).cuda()
    with torch.no_grad():
        output = model(image_batch, t)

    assert output.shape == image_batch.shape


def test_train_step(batch_size: int = 16, image_size: int = 256) -> None:
    model = UNet(3).train().cuda()
    t = torch.rand(batch_size, 1).cuda()
    image_batch = torch.randn(batch_size, 3, image_size, image_size, dtype=torch.float32).cuda()
    optim = Adam(model.parameters())
    for _ in range(5):
        optim.zero_grad()
        output = model(image_batch, t)
        output.sum().backward()
        optim.step()


def test_sampling():
    device = 'cuda:0'
    model = UNet(3).train().to(device)
    samples = sample(16, model, device)
    
    assert samples.shape == (16, 256, 256, 3)
    assert samples.dtype == np.uint8