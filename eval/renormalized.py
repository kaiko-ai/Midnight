"""Wrappers for models that require custom input normalization."""

import timm
import torch
import torch.nn as nn
import torchvision
from typing_extensions import override

from object_tools import ModelBuilder


class Renormalize:
    """Changes the normalization of the images.

    E.g., Renormalize(old={'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]},
                      new={'mean': [0.1, 0.1, 0.1], 'std': [0.1, 0.1, 0.1]})
    will renormalize the data from 0.5 to 0.1 mean and std.

    Args:
        old: The normalization that's been applied to the data.
            e.g., old={'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}.
        new: New normalization parameters.
            e.g., new={'mean': [0.1, 0.1, 0.1], 'std': [0.1, 0.1, 0.1]}.
    """

    def __init__(self, old: dict[str, list[float]], new: dict[str, list[float]]) -> None:
        """Initializes the class.

        Args:
            old: A dict with keys `mean` and `std` representing the old normalization.
            new: A dict with keys `mean` and `std` representing the new normalization.
        """
        super().__init__()

        mean = torch.Tensor(old["mean"])
        new_mean = torch.Tensor(new["mean"])

        std = torch.Tensor(old["std"])
        new_std = torch.Tensor(new["std"])

        # x -> y = (x-m)/s -> (y - (-m/s))/(1/s)
        # (y - ((m2-m)/s))/(s2/s)
        # denormalize is the same as the normalization with mean=-1 and sigma=2
        self._renormalize = torchvision.transforms.Normalize(
            (new_mean - mean) / std, new_std / std, inplace=False
        )

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return self._renormalize(tensor)


class RenormalizingModel(nn.Module):
    """Wrapper class for models with custom normalization."""

    def __init__(  # pylint: disable=dangerous-default-value
        self,
        model: nn.Module,
        new_normalization: dict[str, list[float]],
        data_normalization: dict[str, list[float]] = {
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
        },
    ) -> None:
        """Initializes the model.

        Args:
            data_normalization: The normalization that has already been applied to the input data.
            new_normalization: The desired normalization.
        """
        super().__init__()

        self._renormalize = Renormalize(old=data_normalization, new=new_normalization)

        if isinstance(model, dict) and {"path", "arguments"}.issubset(model):
            if model["arguments"].get("act_layer") == "torch.nn.SiLU":
                model["arguments"]["act_layer"] = torch.nn.SiLU
            if model["arguments"].get("mlp_layer") == "timm.layers.SwiGLUPacked":
                model["arguments"]["mlp_layer"] = timm.layers.SwiGLUPacked
            self._model = ModelBuilder(**model).build()
        else:
            self._model = model

    @override
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""

        return self._model(self._renormalize(tensor))

    @override
    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: int | tuple[int, ...] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        norm=True,
    ) -> tuple[torch.Tensor | tuple[torch.Tensor]]:
        """Returns the intermediate layers of the model."""
        return self._model.get_intermediate_layers(
            self._renormalize(x),
            n=n,
            reshape=reshape,
            return_class_token=return_class_token,
            norm=norm,
        )
