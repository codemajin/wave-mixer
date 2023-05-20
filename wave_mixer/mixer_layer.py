# -*- coding: utf-8 -*-

from typing import Callable, Tuple, Union
import torch
from torch import nn
from torchinfo import summary


class MixerLayer(nn.Module):
    r"""Mixer-architecture layer.

    Examples::

        >>> layer = MixerLayer(num_features=16, channels=8, dropout=0.2, activation='gelu')
        >>> input = torch.randn(32, 8, 16)
        >>> output = layer(input)
            torch.Size([32, 8, 16])
    """

    def __init__(self, num_features: int, channels: int, dropout: float, activation: str, channel_factor: int = 4, token_factor: float = 0.5) -> None:
        """Construct mixer-layer according to input parameters.

        Args:
            num_features (int): the dimension of the network model.
            channels (int): tnumber of channels in the input sample.
            dropout (float): dropout ratio.
            activation (str): the activation function of the intermediate layer, can be a string ("relu" or "gelu" or "tanh").
            channel_factor (int, optional): the ratio of the inner dimension of channel-mixing to input dimension. Defaults to 4.
            token_factor (float, optional): the ratio of the inner dimension of token-mixing to input dimension. Defaults to 0.5.
        """
        super().__init__()
        self.channel_mixing = _SkipConnection(
            num_features=num_features,
            layer=_MLPLayer(
                num_features=channels,
                expansion_factor=channel_factor,
                dropout=dropout,
                dense="conv1d",
                activation=activation
            )
        )

        self.token_mixing = _SkipConnection(
            num_features=num_features,
            layer=_MLPLayer(
                num_features=num_features,
                expansion_factor=token_factor,
                dropout=dropout,
                dense="linear",
                activation=activation
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Handle forward propagation.

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: output tensor.
        """
        y = self.channel_mixing(x)
        y = self.token_mixing(y)
        return y

    def summary(self, input_size: Tuple[int, int, int]) -> None:
        """Summarize this model.

        Args:
            input_size (Tuple[int, int, int]): input shape of network.
        """
        summary(self, input_size=input_size, depth=6, col_names=["input_size", "output_size", "num_params"])
        print("")


class _SkipConnection(nn.Module):
    r"""Skip-connection with layer nomarlization.

    Examples::

        >>> layer = nn.Linear(in_features=16, out_features=16, bias=True)
        >>> skip = _SkipConnection(num_features=16, layer=layer)
        >>> input = torch.randn(32, 8, 16)
        >>> output = layer(input)
            torch.Size([32, 8, 16])
    """

    def __init__(self, num_features: int, layer: Callable) -> None:
        """Construct skip-connection layers according to input parameters.

        Args:
            num_features (int): the dimension of the network model.
            layer (Callable): arbitrary layer of neural network.
        """
        super().__init__()
        self.layer = layer
        self.norm = nn.LayerNorm(normalized_shape=num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Handle forward propagation.

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: output tensor.
        """
        return self.layer(self.norm(x)) + x


class _MLPLayer(nn.Module):
    r"""Consisting of two fully-connected layers and an activation.

    Examples::

        1. Case of 'linear'

        >>> layer = _MLPLayer(num_features=128, expansion_factor=2, dropout=0.2, dense='linear', activation='gelu')
        >>> input = torch.randn(16, 8, 128)
        >>> output = layer(input)
            torch.Size([16, 8, 128])

        2. Case of 'conv1d'

        >>> layer = _MLPLayer(num_features=8, expansion_factor=2, dropout=0.2, dense='conv1d', activation='gelu')
        >>> input = torch.randn(16, 8, 128)
        >>> output = layer(input)
            torch.Size([16, 8, 128])
    """

    def __init__(self, num_features: int, expansion_factor: Union[int, float] = 4, dropout: float = 0.0, dense: str = "linear", activation: str = "gelu") -> None:
        """Construct fully-connected layers according to input parameters.

        Args:
            num_features (int): the dimension of the feedforward network model.
            expansion_factor (Union[int, float], optional): inner dimension expansion ratio to input dimension. Defaults to 4.
            dropout (float, optional): dropout ratio. Defaults to 0.0.
            dense (str, optional): the type of dese layer, can be a string ("linear" or "conv1d"). Defaults to "linear".
            activation (str, optional): the activation function of the intermediate layer, can be a string ("relu" or "gelu" or "tanh"). Defaults to "gelu". Defaults to "gelu".
        """
        super().__init__()
        inner_features = int(num_features * expansion_factor)
        self.network = nn.Sequential(
            self.__dense_layer(dense, num_features, inner_features),
            self.__activation_layer(activation),
            nn.Dropout(p=dropout),

            self.__dense_layer(dense, inner_features, num_features),
            nn.Dropout(p=dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Handle forward propagation.

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: output tensor.
        """
        return self.network(x)

    def __dense_layer(self, dense: str, in_features: int, out_features: int) -> Union[nn.Conv1d, nn.Linear]:
        """Construct one fully-connected layer.

        Args:
            dense (str): the type of dese layer, can be a string ("linear" or "conv1d").
            in_features (int): the input dimension.
            out_features (int): the output dimension.

        Returns:
            Union[nn.Conv1d, nn.Linear]: fully-connected layer ('Conv1d' or 'Linear').
        """
        assert dense in ["conv1d", "linear"], "Parameter 'dense' must be 'conv1d' or 'linear'."

        if dense == "conv1d":
            return nn.Conv1d(in_channels=in_features, out_channels=out_features, kernel_size=1, padding=0, stride=1, bias=True)
        else:
            return nn.Linear(in_features=in_features, out_features=out_features, bias=True)

    def __activation_layer(self, activation: str) -> Union[nn.ReLU, nn.GELU, nn.Tanh]:
        """Select an activation function.

        Args:
            activation (str): the activation function of the intermediate layer, can be a string ("relu" or "gelu" or "tanh"). Defaults to "gelu".

        Returns:
            Union[nn.ReLU, nn.GELU, nn.Tanh]: activation function ("ReLU" or "GELU" or "Tanh").
        """
        supported_activations = {
            "relu": nn.ReLU(inplace=True),
            "tanh": nn.Tanh(),
            "gelu": nn.GELU(),
        }

        assert activation in supported_activations.keys(), "Parameter 'activation' must be 'relu', 'tanh', or 'gelu'."
        return supported_activations[activation]
