from pathlib import Path

import click
import cv2
import torch

from torch import Tensor
from torchdiffeq import odeint

from src import UNet


@torch.no_grad()
@click.command()
@click.argument('num_steps', type=int)
@click.argument('device', type=str, default='cuda:0')
def main(num_steps: int, device: str) -> None:
    unet = UNet(3).eval().to(device)
    unet.load_state_dict(torch.load('checkpoints/model.pt'))

    z = torch.randn(1, 3, 256, 256, dtype=torch.float32, device=device)
    t = torch.linspace(0, 1, num_steps, dtype=torch.float32, device=device)

    def odefunc(t: Tensor, x: Tensor) -> Tensor:
        return unet(
            x,
            torch.full((x.shape[0], 1), t, device=device)
        )

    states = odeint(odefunc, z, t, atol=1e-4, method='dopri5')
    image = torch.cat(list(states), dim=-1)
    image = (255 * image[0].permute(1, 2, 0).cpu().numpy().clip(0, 1)).astype('uint8')
    cv2.imwrite('big-image.png', image)


if __name__ == '__main__':
    main()