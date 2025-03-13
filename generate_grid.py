import math

from pathlib import Path

import click
import cv2
import numpy as np
import torch

from tqdm import trange
from src import UNet, sample


@click.command()
@click.argument('num_samples', type=int)
@click.argument('device', type=str, default='cuda:0')
@click.argument('batch_size', type=int, default=16)
def main(num_samples: int, device: str, batch_size: int) -> None:
    unet = UNet(3).eval().to(device)
    unet.load_state_dict(torch.load('checkpoints/model.pt'))

    side = round(math.sqrt(num_samples))
    result = np.zeros((side * 256, side * 256, 3), dtype=np.uint8)

    for start in trange(0, num_samples, batch_size):
        ns = min(num_samples - start, batch_size)
        generated = sample(
            ns, unet, device,
            tol=1e-4, method='dopri5'
        )

        for idx, generated_sample in enumerate(generated):
            image = generated_sample[..., ::-1]
            i, j = (start + idx) // side, (start + idx) % side
            result[256 * i:256 * i+256, 256 * j:256 * j+256] = image

    cv2.imwrite('grid.png', result)


if __name__ == '__main__':
    main()