"""
Note that image feature is provided by MineCLIP.
"""
import torch
import torch.nn as nn

from mineclip.utils import build_mlp
import mineclip.utils as U

class DummyImgFeat(nn.Module):
    def __init__(self, *, output_dim: int = 512, image_model: str, device: torch.device):
        super().__init__()
        self._output_dim = output_dim
        self._device = device

        if image_model == 'mineclip':
            self._mlp = nn.Identity()
            self._output_dim = 512
        
        else:
            _mlp = build_mlp(
                input_dim=900,
                hidden_dim=128,
                hidden_depth=2,
                output_dim=2
            )
            self._mlp = nn.Sequential(
                nn.Linear(in_features=900, out_features=128),
                nn.ReLU(),
                nn.Linear(in_features=128, out_features=32),
                nn.ReLU(),
                nn.Linear(in_features=32, out_features=2),
                nn.Flatten()
            )
            self._output_dim = 512
    @property
    def output_dim(self):
        return self._output_dim

    def forward(self, x, **kwargs):
        x = U.any_to_torch_tensor(x, device=self._device)
        return self._mlp(x), None
