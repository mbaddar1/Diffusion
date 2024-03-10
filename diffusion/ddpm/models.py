import logging

import torch
from torch import nn
import numpy as np

logger = logging.getLogger()

ACTIVATIONS = {"GELU": nn.GELU(), "Tanh": nn.Tanh(), "Sigmoid": nn.Sigmoid(), "ReLU": nn.ReLU(),
               "Identity": nn.Identity()}


class PositionalEncoding(nn.Module):
    """The classic positional encoding from the original Attention papers"""

    def __init__(
            self,
            model_dim: int = 128,
            maxlen: int = 1024,
            min_freq: float = 1e-4,
            device: str = "cpu",
            dtype=torch.float32,
    ):
        """
        Args:
            model_dim (int, optional): embedding dimension of each token. Defaults to 128.
            maxlen (int, optional): maximum sequence length. Defaults to 1024.
            min_freq (float, optional): use the magic 1/10,000 value! Defaults to 1e-4.
            device (str, optional): accelerator or nah. Defaults to "cpu".
            dtype (_type_, optional): torch dtype. Defaults to torch.float32.
        """
        super().__init__()
        pos_enc = self._get_pos_enc(model_dim=model_dim, maxlen=maxlen, min_freq=min_freq)
        self.register_buffer(
            "pos_enc", torch.tensor(pos_enc, dtype=dtype, device=device)
        )

    def _get_pos_enc(self, model_dim: int, maxlen: int, min_freq: float):
        position = np.arange(maxlen)
        freqs = min_freq ** (2 * (np.arange(model_dim) // 2) / model_dim)
        pos_enc = position[:, None] * freqs[None]
        pos_enc[:, ::2] = np.cos(pos_enc[:, ::2])
        pos_enc[:, 1::2] = np.sin(pos_enc[:, 1::2])
        return pos_enc

    def forward(self, x):
        return self.pos_enc[x]


class GaussianFourierProjection(nn.Module):
    """Positional encoding for continuum states. Think how to embed
    functional dependence on a real-valued scalar, like f(x) -> f(x, t)
    for some scalar time variable t.

    This creates random Gaussian Fourier features. In fact, Random fourier
    Features have an interesting $N \to \infty$ limit for layer width $N$;
    They become Gaussian Processes!
    """

    def __init__(self, embed_dim: int, scale: float = 1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.scale = scale
        self.W = torch.randn(self.embed_dim // 2) * self.scale

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * 3.1415927
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class DiscreteTimeResidualBlock(nn.Module):
    """Generic block to learn a nonlinear function f(x, t), where
    t is discrete and x is continuous."""

    def __init__(self, model_dim: int, maxlen: int = 512, with_time_emb: bool = True, activation_name: str = "GELU"):
        super().__init__()
        self.model_dim = model_dim
        self.emb = PositionalEncoding(model_dim=model_dim, maxlen=maxlen)
        self.lin1 = nn.Linear(model_dim, model_dim)
        self.lin2 = nn.Linear(model_dim, model_dim)
        self.norm = nn.LayerNorm(model_dim)
        # What is GELU activation function
        # 1. https://paperswithcode.com/method/gelu
        # 2. https://arxiv.org/abs/1606.08415v5
        assert activation_name in ACTIVATIONS.keys(), f"activation_name param must be one of {ACTIVATIONS.keys()}"
        self.activation = ACTIVATIONS[activation_name]
        self.with_time_emb = with_time_emb

        logger.info(f"with_time_embedding ? {with_time_emb}")
        logger.info(f"Activation class instance type : {type(self.activation)}")

    def forward(self, x, t):

        if self.with_time_emb:
            t_emb = self.emb(t)
            x_input = x + t_emb
        else:
            x_input = x
        return self.norm(x + self.lin2(self.activation(self.lin1(x_input))))


class BasicDiscreteTimeModel(nn.Module):
    def __init__(self, model_dim: int = 128, data_dim: int = 2, num_resnet_layers: int = 2, with_time_emb: bool = True,
                 activation_name="GELU"):
        super().__init__()
        self.model_dim = model_dim
        self.n_layers = num_resnet_layers
        self.lin_in = nn.Linear(data_dim, model_dim)
        self.lin_out = nn.Linear(model_dim, data_dim)
        self.blocks = nn.ModuleList(
            [DiscreteTimeResidualBlock(model_dim=model_dim, with_time_emb=with_time_emb,
                                       activation_name=activation_name)
             for _ in
             range(num_resnet_layers)]
        )

    def forward(self, x, t):
        x = self.lin_in(x)
        for block in self.blocks:
            x = block(x, t)
        return self.lin_out(x)


class NaiveNeuralNetworkNoiseModel(nn.Module):
    def __init__(self, time_steps: int, hidden_dim: int = 64, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_steps = time_steps
        self.model = nn.Sequential(nn.Linear(2 + 1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        t_norm = (t / float(self.time_steps)).reshape(-1, 1)
        x_aug = torch.cat([x, t_norm], dim=1)
        out = self.model(x_aug)
        return out
