import torch
import torch.nn as nn
from .__config__ import ModelArgsV1
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class NSEA(nn.Module):
    """
    NSEA Block
    """
    def __init__(self, args: ModelArgsV1):
        super(NSEA, self).__init__()
        self.Wx = nn.Linear(args.dim , args.n_heads)
        self.Lx = nn.Linear(args.n_heads, args.dim, bias=False)
    
    def exponential_head(self, matrix_product: torch.Tensor) -> torch.Tensor:
        """
        Exponential head for Non square exponential attention.
        :param matrix_product: Matrix product tensor
        :return: Exponential of the matrix product
        """
        s = torch.eye(matrix_product.size(-2,), matrix_product.size(-1))
        mp1 = matrix_product.clone()
        mp2 = mp1.sum(dim=0, keepdim=True) * mp1
        mp3 = mp2.sum(dim=0, keepdim=True) * mp1
        mp4 = mp3.sum(dim=0, keepdim=True) * mp1
        s = s + mp1 + \
            (1.0 /  2.0) * mp2 + \
            (1.0 /  6.0) * mp3 + \
            (1.0 / 24.0) * mp4
        return s

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through NSEA block.
        :param x: Input tensor of shape (Batch, Time window,
                        No. Features)
        :return: Output tensor after applying NSEA
        """
        # x: Batch, No. Assets, Time window, No. Features
        Wx = self.Wx(x)
        Wx2 = Wx.sum(dim=0, keepdim=True) * Wx
        Lx = self.Lx(Wx2)
        Ex = self.exponential_head(Lx)
        return Ex


class Block(nn.Module):
    """
    Transformer model for language modeling.
    """
    def __init__(self, args: ModelArgsV1):
        super(Block, self).__init__()
        self.args = args
        self.norm = nn.RMSNorm(args.dim)
        self.attn = NSEA(args)
        self.ffnn = nn.Linear(args.dim, args.dim)

    def forward(self, x):
        """
        Forward pass through the block.
        :param x: Input tensor of shape
        """
        x = x + self.attn(self.norm(x))
        x = x + self.ffnn(self.norm(x))
        return x


class RomuloModel(BaseFeaturesExtractor):
    """
    Genesis Model V1: Romulo
    """
    def __init__(self,  observation_space: spaces.Box, *, args: ModelArgsV1, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        self.embd = nn.Linear(args.n_features, args.dim)
        self.blocks = nn.ModuleList([Block(args) for _ in range(args.n_layers)])
        self.norm = nn.RMSNorm(args.dim)
        self.flatten = nn.Flatten(start_dim=1)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Romulo model.
        :param observations: Input tensor of shape (Batch, Time window, No. Features)
        :return: Output tensor of shape (Batch, No. Aactions)
        """
        x = self.embd(observations)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = self.flatten(x)
        return x
