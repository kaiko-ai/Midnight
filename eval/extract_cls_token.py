"""Defines the CLS token extractors."""

import math

import torch
from transformers import modeling_outputs
from typing_extensions import override


class ExtractCLSToken:
    """Extracts the CLS token from a ViT model output."""
    def __call__(
        self, tensor: torch.Tensor | modeling_outputs.BaseModelOutputWithPooling
    ) -> torch.Tensor:
        """Call method for the transformation.

        Args:
            tensor: The tensor representing the model output.
        """
        if isinstance(tensor, torch.Tensor):
            return tensor[:, 0, :]
        if isinstance(tensor, modeling_outputs.BaseModelOutputWithPooling):
            return tensor.last_hidden_state[:, 0, :]
        raise ValueError(f"Unsupported type {type(tensor)}")


class ExtractConcatToken:
    """Extracts the CLS with Mean Patch tokens from a ViT model output."""

    def __init__(self, num_reg_tokens: int = 0) -> None:
        self.num_reg_tokens = num_reg_tokens

    @override
    def __call__(
        self, tensor: torch.Tensor | modeling_outputs.BaseModelOutputWithPooling
    ) -> torch.Tensor:
        """Call method for the transformation.

        Args:
            tensor: The tensor representing the model output.
        """
        if isinstance(tensor, torch.Tensor):
            return torch.cat(
                [tensor[:, 0, :], tensor[:, 1 + self.num_reg_tokens :, :].mean(1)], dim=-1
            )
        if isinstance(tensor, modeling_outputs.BaseModelOutputWithPooling):
            return torch.cat(
                [
                    tensor.last_hidden_state[:, 0, :],
                    tensor.last_hidden_state[:, 1 + self.num_reg_tokens :, :].mean(1),
                ],
                dim=-1,
            )
        raise ValueError(f"Unsupported type {type(tensor)}")


class ExtractPatchFeatures:
    """Extracts the patch features from a ViT model output."""

    def __init__(
        self,
        has_cls_token: bool = True,
        num_reg_tokens: int = 0,
        ignore_remaining_dims: bool = False,
    ) -> None:
        """Initializes the transformation.

        Args:
            has_cls_token: If set to `True`, the model output is expected to have
                a classification token.
            num_reg_tokens: The number of register tokens in the model output.
            ignore_remaining_dims: If set to `True`, ignore the remaining dimensions
                of the patch grid if it is not a square number.
        """
        self._has_cls_token = has_cls_token
        self._num_reg_tokens = num_reg_tokens
        self._ignore_remaining_dims = ignore_remaining_dims

    def __call__(
        self, tensor: torch.Tensor | modeling_outputs.BaseModelOutputWithPooling
    ) -> list[torch.Tensor]:
        """Call method for the transformation.

        Args:
            tensor: The raw embeddings of the model.

        Returns:
            A tensor (batch_size, hidden_size, n_patches_height, n_patches_width)
            representing the model output.
        """
        num_skip = int(self._has_cls_token) + self._num_reg_tokens
        if isinstance(tensor, modeling_outputs.BaseModelOutputWithPooling):
            features = tensor.last_hidden_state[:, num_skip:, :].permute(0, 2, 1)
        else:
            features = tensor[:, num_skip:, :].permute(0, 2, 1)

        batch_size, hidden_size, patch_grid = features.shape
        height = width = int(math.sqrt(patch_grid))
        if height * width != patch_grid:
            if self._ignore_remaining_dims:
                features = features[:, :, -height * width :]
            else:
                raise ValueError(f"Patch grid size must be a square number {patch_grid}.")
        patch_embeddings = features.view(batch_size, hidden_size, height, width)

        return [patch_embeddings]
