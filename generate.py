from pathlib import Path

import click
import cv2
import torch

from tqdm import trange

from src import UNet, sample


@click.command()
@click.argument('num_samples', type=int)
@click.argument('device', type=str, default='cuda:0')
@click.argument('batch_size', type=int, default=16)
def main(num_samples: int, device: str, batch_size: int) -> None:
    unet = UNet(3).eval().to(device)
    unet.load_state_dict(torch.load('checkpoints/model1.pt'))

    for start in trange(0, num_samples, batch_size):
        ns = min(num_samples - start, batch_size)
        generated = sample(
            ns, unet, device,
            tol=1e-4, method='dopri5'
        )

        output_path = Path('samples')
        for idx, generated_sample in enumerate(generated):
            save_path = str(output_path / f'{start + idx}.png')
            image = generated_sample[..., ::-1]
            cv2.imwrite(str(save_path), image)


if __name__ == '__main__':
    main()