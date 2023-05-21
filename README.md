# WaveMixer for PyTorch

Mixer-architecture network applying **MLP-Mixer** to one-dimensional tensor.

MLPMixer's paper: [MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/abs/2105.01601)

## Install

```
$ git clone https://github.com/codemajin/wave-mixer
$ cd wave-mixer
$ pip install .
```

## Usage

For classification, set `out_features` to the number of classes as follows:

```python
import torch
from wave_mixer import WaveMixer

model = WaveMixer(
    in_features=16000,
    out_features=100,
    channels=1,
    patch_size=160,
    mixer_dim=256,
    num_mixers=6
)

input = torch.randn(32, 1, 16000)
output = model(input)   # torch.Size([32, 100])
```

On the other hand, for regression, set `out_features` to **1**.

```python
import torch
from wave_mixer import WaveMixer

model = WaveMixer(
    in_features=16000,
    out_features=1,
    channels=1,
    patch_size=160,
    mixer_dim=256,
    num_mixers=6
)

input = torch.randn(32, 1, 16000)
output = model(input)   # torch.Size([32, 1])
```

`dropout` and `activation` are configurable, too.

```python
model = WaveMixer(
    ...
    dropout=0.2,       # Default to 0.0.
    activation="relu"  # Default to 'gelu', must be 'gelu', 'relu', or 'tanh'.
)
```