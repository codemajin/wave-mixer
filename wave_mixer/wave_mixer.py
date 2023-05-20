# -*- coding: utf-8 -*-

from typing import Tuple

import torch
from einops.layers.torch import Rearrange, Reduce
from torch import nn
from torchinfo import summary

from wave_mixer.mixer_layer import MixerLayer


class WaveMixer(nn.Module):
    r"""Mixer-architecture network applying MLPMixer to one-dimensional tensor.

    Examples::

        >>> mixer = WaveMixer(in_features=24000, out_features=100, channels=1, patch_size=160, mixer_dim=256, num_mixers=6)
        >>> input = torch.randn(32, 1, 24000)
        >>> output = mixer(input)
            torch.Size([32, 100])
    """

    def __init__(self, in_features: int, out_features: int, channels: int, patch_size: int, mixer_dim: int, num_mixers: int,
                 activation: str = "gelu", dropout: float = 0.0) -> None:
        """Construct the WaveMixer according to input parameters.

        Args:
            in_features (int): size of each input sample
            out_features (int): size of each output sample
            channels (int): number of channels in the input sample.
            patch_size (int): patch size.
            mixer_dim (int): the dimension of the mixer layer.
            num_mixers (int): the number of the mixer layer.
            activation (str, optional): the activation function of the intermediate layer, can be a string ("relu" or "gelu" or "tanh"). Defaults to "gelu".
            dropout (float, optional): probability of an element to be zeroed. Defaults to 0.0.
        """
        super().__init__()
        assert (in_features % patch_size) == 0, "Wave length must be divisible by patch size."
        num_patches = in_features // patch_size

        self.network = nn.Sequential(
            Rearrange('b c (l p) -> b l (p c)', p=patch_size),
            nn.Linear(in_features=patch_size * channels, out_features=mixer_dim, bias=True),
            *[MixerLayer(num_features=mixer_dim, channels=num_patches, dropout=dropout, activation=activation) for _ in range(num_mixers)],
            nn.LayerNorm(normalized_shape=mixer_dim),
            Reduce("b n c -> b c", "mean"),
            nn.Linear(in_features=mixer_dim, out_features=out_features, bias=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Handle forward propagation.

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: output tensor.
        """
        return self.network(x)

    def summary(self, input_size: Tuple[int, int, int]) -> None:
        """Summarize this model.

        Args:
            input_size (Tuple[int, int, int]): input shape of network.
        """
        summary(self, input_size=input_size, depth=6, col_names=["input_size", "output_size", "num_params"])
        print("")
