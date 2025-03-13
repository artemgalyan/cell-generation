import logging
import os

from pathlib import Path

import click
import numpy as np
import torch
import torch.nn.functional as F
import wandb

from torch import Tensor
from torch.optim import Optimizer, Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from tqdm.auto import tqdm, trange

from src import UNet, HistologyDataset, sample


def schedule(epoch: int) -> float:
    print(epoch)
    if epoch < 50:
        return 1e-3
    if epoch < 100:
        return 1e-4

    return 1e-5


def train_epoch(
    model: UNet,
    optimizer: Optimizer,
    loader: DataLoader,
    device: str,
    accumulation_steps: int = 1,
    sigma_min: float = 1e-4,
    verbose: bool = False,
) -> float:
    model.train()
    optimizer.zero_grad()
    iterable = loader
    if verbose:
        iterable = tqdm(iterable, leave=False)
    
    running_loss = None
    
    for step, x1 in enumerate(iterable):
        t = torch.rand(len(x1), 1, device=device, dtype=torch.float32)
        x1 = x1.to(device)
        x0 = torch.randn_like(x1)
        
        tt = t.reshape(-1, 1, 1, 1)
        xt = tt * x1 + (1 - (1 - sigma_min) * tt) * x0
        flow = (x1 - (1 - sigma_min) * xt) / (1 - (1 - sigma_min) * tt)
        preds = model(xt, t)
        loss = F.mse_loss(flow, preds)
        (loss / accumulation_steps).backward()
        if running_loss is None:
            running_loss = float(loss.detach().cpu().item())
        running_loss = 0.9 * running_loss + 0.1 * float(loss.detach().cpu().item())
        if step % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        if verbose:
            iterable.set_description(f'Loss: {running_loss}')

    optimizer.zero_grad()
    return running_loss


@click.command()
@click.argument('num_epochs', type=int)
@click.option('-d', '--device', type=str, default='cuda:0')
@click.option('-bs', '--batch_size', type=int, default=16)
@click.option('-nw', '--num_workers', type=int, default=4)
@click.option('-s', '--sigma', type=float, default=1e-4)
@click.option('-v', '--verbose', is_flag=True)
def main(
    num_epochs: int,
    device: str,
    batch_size: int,
    num_workers: int,
    sigma: float,
    verbose: bool
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
    )

    run = wandb.init(
        project='cell-generation',
        config={
            'sigma': sigma
        }
    )

    logging.info(f'Running from {os.getcwd()}')
    print(os.getcwd())

    dataset = HistologyDataset(
        images=list(Path('data/train').glob('**/*.png'))
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        pin_memory_device=device
    )

    logging.info(f'Loaded data. Number of training images: {len(dataset)}. Number of batches: {len(dataloader)}')


    model = UNet(3).train().to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    optimizer = Adam(model.parameters(), lr=1e-4)
    # scheduler = LambdaLR(optimizer, schedule)
    # scheduler.step()

    logging.info(f'Number of model paramters: {parameter_count}')

    iterable = range(num_epochs)
    if verbose:
        iterable = trange(num_epochs)
    
    for epoch in iterable:
        epoch_loss = train_epoch(
            model, optimizer, dataloader,
            device, verbose=verbose,
            accumulation_steps=2,
        )
        # scheduler.step()

        if verbose:
            iterable.set_description(f'Loss: {epoch_loss}')
        
        try:
            samples = sample(16, model, device)
        except:
            samples = np.zeros((16, 256, 256, 3), dtype=np.uint8)
        grid = np.zeros((4 * 256, 4 * 256, 3), dtype=np.uint8)
        for i in range(4):
            for j in range(4):
                grid[i * 256:(i + 1) * 256, j * 256:(j + 1) * 256] = samples[4 * i + j]

        run.log({
            'loss': epoch_loss,
            # 'lr': scheduler.get_last_lr()[0],
            'generated_images': wandb.Image(grid, caption='Generated images')
        })
    
    torch.save(model.state_dict(), 'checkpoints/model.pt')


if __name__ == '__main__':
    main()