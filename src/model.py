import torch
import torch.nn.functional as F

from torch import nn, Tensor


class UpsampleBlock(nn.Sequential):
    def __init__(self, dim: int) -> None:
        super().__init__(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(dim, dim // 2, kernel_size=3, padding=1)
        )


class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-5)
        return self.gamma * (x * Nx) + self.beta + x
    

class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.register_buffer('freqs', torch.arange(1, dim // 2 + 1) * torch.pi)
        self.weight = nn.Parameter(torch.randn(dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = self.freqs * t.unsqueeze(-1)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1) * self.weight
        return emb.reshape(t.shape[0], self.dim, 1, 1)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        expansion: int = 4,
        activation: type[nn.Module] = nn.GELU,
        norm: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.interm_channels = dim * expansion
        self.expansion = expansion
        self.activation_cls = activation
        self.norm_cls = norm

        self.time_embedding = TimeEmbedding(dim)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = norm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = activation()
        self.norm2 = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        z = x
        x = self.dwconv(x) + self.time_embedding(t)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.norm2(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        return x + z


class EncoderBlock(nn.Module):
    def __init__(self, dim: int, num_blocks: int = 2) -> None:
        super().__init__()

        self.dim = dim
        self.num_blocks = num_blocks

        self.res_blocks = nn.ModuleList([
            ResidualBlock(dim) for _ in range(num_blocks)
        ])

        self.downsample = nn.Conv2d(
            dim, 2 * dim, kernel_size=4, stride=2, padding=1
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        for block in self.res_blocks:
            x = block(x, t)
        return x, self.downsample(x)
    

class DecoderBlock(nn.Module):
    def __init__(self, dim: int, num_blocks: int = 2) -> None:
        super().__init__()

        self.dim = dim
        self.num_blocks = num_blocks

        self.project = nn.Conv2d(2 * dim, dim, kernel_size=1)

        self.res_blocks = nn.Sequential(*[
            ResidualBlock(dim) for _ in range(num_blocks)
        ])

        self.upsample = UpsampleBlock(dim)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        x = self.project(x)
        for block in self.res_blocks:
            x = block(x, t)
        return self.upsample(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        dim: int = 16,
        height: int = 4,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.dim = dim
        self.height = height

        self.stem_conv = nn.Conv2d(self.in_channels, self.dim, kernel_size=3, padding=1)
        self.stem_block = EncoderBlock(self.dim)

        self.encoder = nn.ModuleList([
            EncoderBlock(self.dim * 2 ** (i + 1))
            for i in range(self.height)
        ])

        bottom_dim = self.dim * 2 ** (self.height + 1)
        self.time_embedding = TimeEmbedding(bottom_dim)
        self.bottom = nn.Sequential(
            nn.Conv2d(bottom_dim, bottom_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(bottom_dim, bottom_dim, kernel_size=1),
            nn.GELU(),
            UpsampleBlock(bottom_dim),
            # nn.ConvTranspose2d(bottom_dim, bottom_dim // 2, kernel_size=4, stride=2, padding=1)
        )

        self.decoder = nn.ModuleList([
            DecoderBlock(self.dim * 2 ** (i + 1))
            for i in reversed(range(self.height))
        ])

        self.out = nn.Sequential(
            nn.Conv2d(2 * self.dim, self.in_channels, kernel_size=1),
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        x = self.stem_conv(x)
        inputs, x = self.stem_block(x, t)
        outputs = []
        for encoder_block in self.encoder:
            out, x = encoder_block(x, t)
            outputs.append(out)

        x = self.bottom(x + self.time_embedding(t))
        for decoder_block, y in zip(self.decoder, reversed(outputs)):
            z = torch.cat([x, y], dim=1)
            x = decoder_block(z, t)

        return self.out(torch.cat([x, inputs], dim=1))
