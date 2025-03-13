import torch

from numpy.typing import NDArray
from torch import Tensor
from torchdiffeq import odeint

from .model import UNet


@torch.no_grad()
def sample(n: int, model: UNet, device: str, num_steps: int = 2, tol: float = 1e-4, **kw) -> NDArray:
    model.eval()
    z = torch.randn(n, 3, 256, 256, dtype=torch.float32, device=device)
    t = torch.linspace(0, 1, num_steps, dtype=torch.float32, device=device)

    def odefunc(t: Tensor, x: Tensor) -> Tensor:
        return model(
            x,
            torch.full((x.shape[0], 1), t, device=device)
        )

    states = odeint(odefunc, z, t, atol=tol, **kw)
    samples = states[-1]
    return (255 * samples.cpu().permute(0, 2, 3, 1).numpy().clip(0, 1)).astype('uint8')
