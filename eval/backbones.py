"""Wrappers for custom models."""

import timm
import torch
import torch.nn as nn
import torchvision
from typing_extensions import override

from object_tools import load_model_checkpoint


from typing import Any, Callable

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from typing_extensions import override

from extract_cls_token import ExtractCLSToken


class HuggingFaceModel(nn.Module):
    """Wrapper class for loading HuggingFace `transformers` models."""

    def __init__(
        self,
        model_name_or_path: str,
        output_transform: Callable = ExtractCLSToken(),
        with_config: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initializes the model.

        Args:
            model_name_or_path: The model name or path to load the model from.
                This can be a local path or a model name from the `HuggingFace`
                model hub.
            output_transform: The transform to apply to the output tensor produced by the model.
        """
        super().__init__()

        self._output_transform = output_transform

        config = AutoConfig.from_pretrained(model_name_or_path) if with_config else None
        self._model = AutoModel.from_pretrained(model_name_or_path, config=config, **kwargs)

    @override
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""

        tensor = self._model(tensor)
        return self._output_transform(tensor)


class TimmModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        concat_mean_patch_tokens: bool = False,
        **kwargs,
    ):
        super().__init__()
        if kwargs.get("mlp_layer") == "timm.layers.SwiGLUPacked":
            kwargs["mlp_layer"] = timm.layers.SwiGLUPacked
        if kwargs.get("act_layer") == "torch.nn.SiLU":
            kwargs["act_layer"] = torch.nn.SiLU
        self.model = timm.create_model(model_name, **kwargs)
        self.concat_mean_patch_tokens = concat_mean_patch_tokens
        self.out_indices = kwargs.get("out_indices")

    @override
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""

        if self.out_indices is not None:
            return self.model(tensor)

        output = self.model.forward_features(tensor)

        class_token = output[:, 0]
        patch_tokens = output[:, self.model.num_prefix_tokens :]  # skip cls token and registers

        if self.concat_mean_patch_tokens:
            # concatenate class token and average pool of patch tokens
            embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
        else:
            embedding = class_token
        return embedding


# Based on https://huggingface.co/paige-ai/Virchow2
class Virchow2(TimmModel):
    def __init__(self, concat_mean_patch_tokens: bool = True, **kwargs):
        super().__init__(
            model_name="hf-hub:paige-ai/Virchow2",
            pretrained=True,
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=torch.nn.SiLU,
            concat_mean_patch_tokens=concat_mean_patch_tokens,
            **kwargs,
        )
        if self.out_indices is None:
            assert self.model.num_prefix_tokens == 5


class Kaiko(nn.Module):
    """A wrapper constructing custom embeddings for the standard ViT models.
    The final embedding is a concatenation of the class token with the average of the patch tokens.
    """

    def __init__(
        self,
        repo_or_dir: str,
        model: str,
        pretrained: bool | None = None,
        ckpt_path: str | None = None,
        ckpt_submodule: str | None = None,
        concat_mean_patch_tokens: bool = False,
        resize: int | None = None,
        mode: str = "bilinear",
        antialias: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.model = torch.hub.load(
            repo_or_dir=repo_or_dir,
            model=model,
            **({"pretrained": pretrained} if pretrained is not None else {}),
            **kwargs,
        )

        if ckpt_path is not None:
            load_model_checkpoint(self.model, ckpt_path, ckpt_submodule=ckpt_submodule)

        self.concat_mean_patch_tokens = concat_mean_patch_tokens
        self.resize = resize
        self.mode = mode
        self.antialias = antialias
        self.out_indices = kwargs.get("out_indices")

    @override
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        if self.resize and tensor.numel() > 0:
            tensor = torchvision.transforms.functional.center_crop(tensor, min(tensor.shape[-2:]))
            tensor = nn.functional.interpolate(
                tensor, size=(self.resize, self.resize), mode=self.mode, antialias=self.antialias
            )

        if self.out_indices is not None:
            return self.model(tensor)

        out = self.model.forward_features(tensor)

        if isinstance(out, torch.Tensor):
            class_token = out[:, 0]
            patch_tokens = out[:, 1:]
        else:
            class_token = out["x_norm_clstoken"]
            patch_tokens = out["x_norm_patchtokens"]

        if self.concat_mean_patch_tokens:
            # concatenate class token and average pool of patch tokens
            embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
        else:
            embedding = class_token
        return embedding

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: int | tuple[int, ...] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        norm=True,
    ) -> tuple[torch.Tensor | tuple[torch.Tensor]]:
        """Returns the intermediate layers of the model."""
        return self.model.get_intermediate_layers(
            x, n=n, reshape=reshape, return_class_token=return_class_token, norm=norm
        )
