"""Internal Pathology FMs from kaiko.ai."""

from typing import Tuple

import torch
import yaml
from eva.core.models import wrappers
from eva.vision.models.networks.backbones.registry import register_model
from torch import nn
from typing_extensions import override

from object_tools import ModelBuilder


@register_model("pathology/kaiko_vits16")
def kaiko_vits16(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    """Initializes the ViTS-16 pathology FM by kaiko.ai.

    Args:
        dynamic_img_size: Support different input image sizes by allowing to change
            the grid size (interpolate abs and/or ROPE pos) in the forward pass.
        out_indices: Whether and which multi-level patch embeddings to return.

    Returns:
        The model instance.
    """
    return torch.hub.load(  # type: ignore
        repo_or_dir="kaiko-ai/towards_large_pathology_fms",
        model="vits16",
        trust_repo=True,
        dynamic_img_size=dynamic_img_size,
        out_indices=out_indices,
    )


@register_model("pathology/kaiko_vits8")
def kaiko_vits8(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    """Initializes the ViTS-8 pathology FM by kaiko.ai.

    Args:
        dynamic_img_size: Support different input image sizes by allowing to change
            the grid size (interpolate abs and/or ROPE pos) in the forward pass.
        out_indices: Whether and which multi-level patch embeddings to return.

    Returns:
        The model instance.
    """
    return torch.hub.load(  # type: ignore
        repo_or_dir="kaiko-ai/towards_large_pathology_fms",
        model="vits8",
        trust_repo=True,
        dynamic_img_size=dynamic_img_size,
        out_indices=out_indices,
    )


@register_model("pathology/kaiko_vitb16")
def kaiko_vitb16(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    """Initializes the ViTB-16 pathology FM by kaiko.ai.

    Args:
        dynamic_img_size: Support different input image sizes by allowing to change
            the grid size (interpolate abs and/or ROPE pos) in the forward pass.
        out_indices: Whether and which multi-level patch embeddings to return.

    Returns:
        The model instance.
    """
    return torch.hub.load(  # type: ignore
        repo_or_dir="kaiko-ai/towards_large_pathology_fms",
        model="vitb16",
        trust_repo=True,
        dynamic_img_size=dynamic_img_size,
        out_indices=out_indices,
    )


@register_model("pathology/kaiko_vitb8")
def kaiko_vitb8(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    """Initializes the ViTB-8 pathology FM by kaiko.ai.

    Args:
        dynamic_img_size: Support different input image sizes by allowing to change
            the grid size (interpolate abs and/or ROPE pos) in the forward pass.
        out_indices: Whether and which multi-level patch embeddings to return.

    Returns:
        The model instance.
    """
    return torch.hub.load(  # type: ignore
        repo_or_dir="kaiko-ai/towards_large_pathology_fms",
        model="vitb8",
        trust_repo=True,
        dynamic_img_size=dynamic_img_size,
        out_indices=out_indices,
    )


@register_model("pathology/kaiko_vitl14")
def kaiko_vitl14(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    """Initializes the ViTL-14 pathology FM by kaiko.ai.

    Args:
        dynamic_img_size: Support different input image sizes by allowing to change
            the grid size (interpolate abs and/or ROPE pos) in the forward pass.
        out_indices: Whether and which multi-level patch embeddings to return.

    Returns:
        The model instance.
    """
    return torch.hub.load(  # type: ignore
        repo_or_dir="kaiko-ai/towards_large_pathology_fms",
        model="vitl14",
        trust_repo=True,
        dynamic_img_size=dynamic_img_size,
        out_indices=out_indices,
    )


class KaikoModel(wrappers.BaseModel):
    """Model wrapper for `torch.hub` models."""

    def __init__(
        self,
        model_yaml_str: str,
        out_indices: int | Tuple[int, ...] | None = None,
        norm: bool = True,
    ) -> None:
        """Initializes the encoder.

        Args:
            model_yaml_str: Model config in yaml str.
            out_indices: Returns last n blocks if `int`, all if `None`, select
                matching indices if sequence.
        """
        super().__init__()

        self._model_yaml_str = model_yaml_str
        self._out_indices = out_indices
        self._norm = norm
        self.load_model()

    @override
    def load_model(self) -> None:
        """Builds and loads the torch.hub model."""
        self._model: nn.Module = ModelBuilder(**yaml.safe_load(self._model_yaml_str)).build()

    @override
    def model_forward(self, tensor: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        if self._out_indices is None:
            return self._model(tensor)

        return list(
            self._model.get_intermediate_layers(
                tensor,
                self._out_indices,
                reshape=True,
                return_class_token=False,
                norm=self._norm,
            )
        )


@register_model("DINOv2_vitb14_tcga")
def model(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = """
    ckpt_path: /mnt/vast01/shared/experimental/pathology_fm/mikhail/runs/1A83778A-859E-4DE4-9EA1-073A263E5BDD/lightning_logs/version_0/checkpoints/epoch_099-step_166700.ckpt
    ckpt_submodule: teacher.backbone
    path: torch.hub.load
    arguments:
      repo_or_dir: facebookresearch/dinov2:main
      model: dinov2_vitb14
      pretrained: false
    """
    return KaikoModel(model_str, out_indices)


@register_model("DINOv2_vitb14_tcga_SRA-0.2-apply0.5")
def model(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = """
    ckpt_path: /mnt/vast01/shared/experimental/pathology_fm/mikhail/runs/200B3AB5-4439-42C0-BA95-A6314013E75C/lightning_logs/version_0/checkpoints/epoch_099-step_166700.ckpt
    ckpt_submodule: teacher.backbone
    path: torch.hub.load
    arguments:
      repo_or_dir: facebookresearch/dinov2:main
      model: dinov2_vitb14
      pretrained: false
    """
    return KaikoModel(model_str, out_indices)


@register_model("DINOv2_vitb14_tcga_SRA")
def model(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = """
    ckpt_path: /mnt/vast01/shared/experimental/pathology_fm/mikhail/runs/3749129B-B887-4584-99D8-29ECEE3A5A75/lightning_logs/version_0/checkpoints/epoch_099-step_166700.ckpt
    ckpt_submodule: teacher.backbone
    path: torch.hub.load
    arguments:
      repo_or_dir: facebookresearch/dinov2:main
      model: dinov2_vitb14
      pretrained: false
    """
    return KaikoModel(model_str, out_indices)


@register_model("DINOv2_vitb14_tcga+NKI")
def model(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = """
    ckpt_path: /mnt/vast01/shared/experimental/pathology_fm/mikhail/runs/116FF48B-12AE-4212-B33E-9DEF855D42D0/lightning_logs/version_0/checkpoints/epoch_099-step_166700.ckpt
    ckpt_submodule: teacher.backbone
    path: torch.hub.load
    arguments:
      repo_or_dir: facebookresearch/dinov2:main
      model: dinov2_vitb14
      pretrained: false
    """
    return KaikoModel(model_str, out_indices)


@register_model("DINOv2_vitb14_tcga+NKI_300-long")
def model(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = """
    ckpt_path: /mnt/vast01/shared/experimental/pathology_fm/mikhail/runs/D3D61EE3-1027-4722-B9D3-F2FA3EB53939/lightning_logs/version_0/checkpoints/epoch_299-step_500100.ckpt
    ckpt_submodule: teacher.backbone
    path: torch.hub.load
    arguments:
      repo_or_dir: facebookresearch/dinov2:main
      model: dinov2_vitb14
      pretrained: false
    """
    return KaikoModel(model_str, out_indices)


@register_model("DINOv2_vitb14_tcga+NKI_300-long_momentum-0.9985")
def model(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = """
    ckpt_path: /mnt/vast01/shared/experimental/pathology_fm/mikhail/runs/9C0585FF-CEA7-4829-AAFE-01A4EAD0780E/lightning_logs/version_0/checkpoints/epoch_299-step_500100.ckpt
    ckpt_submodule: teacher.backbone
    path: torch.hub.load
    arguments:
      repo_or_dir: facebookresearch/dinov2:main
      model: dinov2_vitb14
      pretrained: false
    """
    return KaikoModel(model_str, out_indices)


@register_model("DINOv2_vitb14_tcga+CPTAC")
def model(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = """
    ckpt_path: /mnt/vast01/shared/experimental/pathology_fm/mikhail/runs/F787A20E-3EF5-47B2-9F3F-FAC8365B5DB8/lightning_logs/version_0/checkpoints/epoch_099-step_166700.ckpt
    ckpt_submodule: teacher.backbone
    path: torch.hub.load
    arguments:
      repo_or_dir: facebookresearch/dinov2:main
      model: dinov2_vitb14
      pretrained: false
    """
    return KaikoModel(model_str, out_indices)


@register_model("DINOv2_vitb14_tcga+GTEx")
def model(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = """
    ckpt_path: /mnt/vast01/shared/experimental/pathology_fm/mikhail/runs/E15DA9D1-AA5B-452E-BFFA-EAF5AD03C618/lightning_logs/version_0/checkpoints/epoch_099-step_166700.ckpt
    ckpt_submodule: teacher.backbone
    path: torch.hub.load
    arguments:
      repo_or_dir: facebookresearch/dinov2:main
      model: dinov2_vitb14
      pretrained: false
    """
    return KaikoModel(model_str, out_indices)


@register_model("DINOv2_vitb14_tcga_RandStainNA")
def model(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = """
    ckpt_path: /mnt/vast01/shared/experimental/pathology_fm/mikhail/runs/B429E466-B181-4B0D-9456-26DF4C40B819/lightning_logs/version_0/checkpoints/epoch_099-step_166700.ckpt
    ckpt_submodule: teacher.backbone
    path: torch.hub.load
    arguments:
      repo_or_dir: facebookresearch/dinov2:main
      model: dinov2_vitb14
      pretrained: false
    """
    return KaikoModel(model_str, out_indices)


@register_model("DINOv2_vitb14_tcga_noHED_noHSV")
def model(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = """
    ckpt_path: /mnt/vast01/shared/experimental/pathology_fm/mikhail/runs/693EE61B-DA87-4725-8338-B145BEF9D495/lightning_logs/version_0/checkpoints/epoch_099-step_166700.ckpt
    ckpt_submodule: teacher.backbone
    path: torch.hub.load
    arguments:
      repo_or_dir: facebookresearch/dinov2:main
      model: dinov2_vitb14
      pretrained: false
    """
    return KaikoModel(model_str, out_indices)


@register_model("DINOv2_vitb14_four")
def DINOv2_vitb14_four(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = """
        ckpt_path: /mnt/vast01/shared/experimental/pathology_fm/mikhail/runs/FM-A9A4C593-4CC0-4844-8896-5D15E9A2052E/epoch_099-step_166700.ckpt
        ckpt_submodule: teacher.backbone
        path: torch.hub.load
        arguments:
          repo_or_dir: facebookresearch/dinov2:main
          model: dinov2_vitb14
          pretrained: false
    """
    return KaikoModel(model_str, out_indices)


@register_model("DINOv2_vitb14_four_restarts")
def DINOv2_vitb14_four(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = """
    ckpt_path: /mnt/vast01/shared/experimental/pathology_fm/mikhail/runs/551D1520-BAD2-4DE7-91F5-F0FFA0EC8925/lightning_logs/version_0/checkpoints/epoch_099-step_166700.ckpt
    ckpt_submodule: teacher.backbone
    path: torch.hub.load
    arguments:
      repo_or_dir: facebookresearch/dinov2:main
      model: dinov2_vitb14
      pretrained: false
    """
    return KaikoModel(model_str, out_indices)


@register_model("DINOv2_vitb14_tcga_KoLeo_noHED")
def DINOv2_vitb14_four(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = """
    ckpt_path: /mnt/vast01/shared/experimental/pathology_fm/mikhail/runs/76953D51-8F71-4201-9F96-7567ACB038BE/lightning_logs/version_0/checkpoints/epoch_099-step_166700.ckpt
    ckpt_submodule: teacher.backbone
    path: torch.hub.load
    arguments:
      repo_or_dir: facebookresearch/dinov2:main
      model: dinov2_vitb14
      pretrained: false
    """
    return KaikoModel(model_str, out_indices)


@register_model("DINOv2_vitb14_tcga_noHED")
def DINOv2_vitb14_four(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = """
    ckpt_path: /mnt/vast01/shared/experimental/pathology_fm/mikhail/runs/99C75885-14C8-4595-B19F-B588896B90B7/lightning_logs/version_0/checkpoints/epoch_099-step_166700.ckpt
    ckpt_submodule: teacher.backbone
    path: torch.hub.load
    arguments:
      repo_or_dir: facebookresearch/dinov2:main
      model: dinov2_vitb14
      pretrained: false
    """
    return KaikoModel(model_str, out_indices)


@register_model("DINOv2_vitb14_four_KoLeo_noHED_centering")
def model(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = """
    ckpt_path: /mnt/vast01/shared/experimental/pathology_fm/mikhail/runs/5E226107-07EE-4612-A492-72DA18FD7C0E/lightning_logs/version_0/checkpoints/epoch_099-step_166700.ckpt
    ckpt_submodule: teacher.backbone
    path: torch.hub.load
    arguments:
      repo_or_dir: facebookresearch/dinov2:main
      model: dinov2_vitb14
      pretrained: false
    """
    return KaikoModel(model_str, out_indices)


@register_model("DINOv2_vitb14_four_noHED")
def model(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = """
    ckpt_path: /mnt/vast01/shared/experimental/pathology_fm/mikhail/runs/0CB3BB8D-E9C3-4BCE-8ABD-6AA1AF2F6979/lightning_logs/version_0/checkpoints/epoch_099-step_166700.ckpt
    ckpt_submodule: teacher.backbone
    path: torch.hub.load
    arguments:
      repo_or_dir: facebookresearch/dinov2:main
      model: dinov2_vitb14
      pretrained: false
    """
    return KaikoModel(model_str, out_indices)


@register_model("DINOv2_vitb14_four_KoLeo")
def model(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = """
    ckpt_path: /mnt/vast01/shared/experimental/pathology_fm/mikhail/runs/A050D239-0686-4704-84EA-4BEDA0C55030/lightning_logs/version_0/checkpoints/epoch_099-step_166700.ckpt
    ckpt_submodule: teacher.backbone
    path: torch.hub.load
    arguments:
      repo_or_dir: facebookresearch/dinov2:main
      model: dinov2_vitb14
      pretrained: false
    """
    return KaikoModel(model_str, out_indices)


@register_model("DINOv2_vitb14_four_KoLeo_noHED")
def model(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = """
    ckpt_path: /mnt/vast01/shared/experimental/pathology_fm/mikhail/runs/B399989A-4309-4676-B457-0AFF9486D5CC/lightning_logs/version_0/checkpoints/epoch_099-step_166700.ckpt
    ckpt_submodule: teacher.backbone
    path: torch.hub.load
    arguments:
      repo_or_dir: facebookresearch/dinov2:main
      model: dinov2_vitb14
      pretrained: false
    """
    return KaikoModel(model_str, out_indices)


@register_model("DINOv2_vitb14_four-TCGA")
def model(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = """
    ckpt_path: /mnt/vast01/shared/experimental/pathology_fm/mikhail/runs/D5ABCEE4-D075-498C-84F5-6B3E28864174/lightning_logs/version_0/checkpoints/epoch_099-step_166700.ckpt
    ckpt_submodule: teacher.backbone
    path: torch.hub.load
    arguments:
      repo_or_dir: facebookresearch/dinov2:main
      model: dinov2_vitb14
      pretrained: false
    """
    return KaikoModel(model_str, out_indices)


@register_model("DINOv2_vitb14_four-GTEx_hsv0.45")
def model(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = """
        ckpt_path: /mnt/vast01/shared/experimental/pathology_fm/mikhail/runs/00F52B7E-1D47-443A-ADB5-B481E5513E23/lightning_logs/version_0/checkpoints/epoch_099-step_166700.ckpt
        ckpt_submodule: teacher.backbone
        path: torch.hub.load
        arguments:
          repo_or_dir: facebookresearch/dinov2:main
          model: dinov2_vitb14
          pretrained: false
    """
    return KaikoModel(model_str, out_indices)


@register_model("DINOv2_vitb14_four-GTEx_hsv0.6")
def model(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = """
    ckpt_path: /mnt/vast01/shared/experimental/pathology_fm/mikhail/runs/59CBA6A7-B2AD-4D9B-8C26-356A0C05BBF7/lightning_logs/version_0/checkpoints/epoch_099-step_166700.ckpt
    ckpt_submodule: teacher.backbone
    path: torch.hub.load
    arguments:
      repo_or_dir: facebookresearch/dinov2:main
      model: dinov2_vitb14
      pretrained: false
    """
    return KaikoModel(model_str, out_indices)


@register_model("DINOv2_vitb14_four_hsv0.6")
def DINOv2_vitb14_four_hsv_06(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = """
        ckpt_path: /mnt/vast01/shared/experimental/pathology_fm/mikhail/runs/FM-7C006271-6762-4D2E-B831-6447B37B1BDD/lightning_logs/version_0/checkpoints/epoch_099-step_166700.ckpt
        ckpt_submodule: teacher.backbone
        path: torch.hub.load
        arguments:
          repo_or_dir: facebookresearch/dinov2:main
          model: dinov2_vitb14
          pretrained: false
    """
    return KaikoModel(model_str, out_indices)


@register_model("DINOv2_vitg14_from_imagenet_tcga-nki_epoch_274")
def DINOv2_vitg14_from_imagenet_tcga_nki_epoch_274(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = """
        path: backbones.Kaiko
        arguments:
          repo_or_dir: facebookresearch/dinov2:main
          model: dinov2_vitg14
          pretrained: false
          concat_mean_patch_tokens: false
          # ckpt_path: az://experimental@stkaikodtpprdlab.blob.core.windows.net/pathology_fm/runs/nebul/vitg14_tcga-nki_epoch_274-step_458425.pth
          ckpt_path: /mnt/vast01/shared/experimental/pathology_fm/mikhail/runs/FM-1738EDAF-99E8-48E1-B1F8-498B280E098F/teacher.backbone/epoch_274-step_458425.pth
    """
    return KaikoModel(model_str, out_indices)


@register_model("DINOv2_vitg14_from_imagenet_tcga-nki_epoch_274_concat")
def DINOv2_vitg14_from_imagenet_tcga_nki_epoch_274_concat(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = """
        path: backbones.Kaiko
        arguments:
          repo_or_dir: facebookresearch/dinov2:main
          model: dinov2_vitg14
          pretrained: false
          concat_mean_patch_tokens: true
          # ckpt_path: az://experimental@stkaikodtpprdlab.blob.core.windows.net/pathology_fm/runs/nebul/vitg14_tcga-nki_epoch_274-step_458425.pth
          ckpt_path: /mnt/vast01/shared/experimental/pathology_fm/mikhail/runs/FM-1738EDAF-99E8-48E1-B1F8-498B280E098F/teacher.backbone/epoch_274-step_458425.pth
    """
    return KaikoModel(model_str, out_indices)


@register_model("vitB14")
def model(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = """
        path: torch.hub.load
        arguments:
          repo_or_dir: facebookresearch/dinov2:main
          model: dinov2_vitb14
          pretrained: false
    """
    return KaikoModel(model_str, out_indices)


@register_model("DINOv2_vitB14_imagenet")
def model(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = """
        path: torch.hub.load
        arguments:
          repo_or_dir: facebookresearch/dinov2:main
          model: dinov2_vitb14
          pretrained: true
    """
    return KaikoModel(model_str, out_indices)


@register_model("DINOv2_vitg14_imagenet")
def DINOv2_vitg14_imagenet(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = """
        path: torch.hub.load
        arguments:
          repo_or_dir: facebookresearch/dinov2:main
          model: dinov2_vitg14
          pretrained: true
    """
    return KaikoModel(model_str, out_indices)


@register_model("DINOv2_vitg14_imagenet_concat")
def vitg14(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = """
        path: backbones.Kaiko
        arguments:
          repo_or_dir: facebookresearch/dinov2:main
          model: dinov2_vitg14
          pretrained: true
          concat_mean_patch_tokens: true
    """
    return KaikoModel(model_str, out_indices)


@register_model("vitg14")
def vitg14(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = """
        path: torch.hub.load
        arguments:
          repo_or_dir: facebookresearch/dinov2:main
          model: dinov2_vitg14
          pretrained: false
    """
    return KaikoModel(model_str, out_indices)


@register_model("vitg14_concat")
def vitg14(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = """
        path: backbones.Kaiko
        arguments:
          repo_or_dir: facebookresearch/dinov2:main
          model: dinov2_vitg14
          pretrained: false
          concat_mean_patch_tokens: true
    """
    return KaikoModel(model_str, out_indices)


@register_model("vitg14_init_1e-5")
def vitg14_init(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = """
        path: torch.hub.load
        arguments:
          repo_or_dir: facebookresearch/dinov2:main
          model: dinov2_vitg14
          pretrained: false
          init_values: 1.0e-5
    """
    return KaikoModel(model_str, out_indices)


@register_model("vitg14_init_0")
def vitg14_init_0(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = """
        path: torch.hub.load
        arguments:
          repo_or_dir: facebookresearch/dinov2:main
          model: dinov2_vitg14
          pretrained: false
          init_values: 0.0
    """
    return KaikoModel(model_str, out_indices)


@register_model("DINOv2_vitg14_from_imagenet_tcga_100M_epoch_294")
def model(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = """
        ckpt_path: /mnt/vast01/shared/experimental/pathology_fm/mikhail/runs/1FDA5ADA-8357-4F5E-8A5C-8758F7751471/teacher.backbone/epoch_294-step_491765.pth
        path: torch.hub.load
        arguments:
          repo_or_dir: facebookresearch/dinov2:main
          model: dinov2_vitg14
          pretrained: false
    """
    return KaikoModel(model_str, out_indices)


@register_model("DINOv2_vitg14_from_imagenet_tcga_100M_epoch_294_concat")
def model(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = """
        path: backbones.Kaiko
        arguments:
          repo_or_dir: facebookresearch/dinov2:main
          model: dinov2_vitg14
          pretrained: false
          concat_mean_patch_tokens: true
          ckpt_path: /mnt/vast01/shared/experimental/pathology_fm/mikhail/runs/1FDA5ADA-8357-4F5E-8A5C-8758F7751471/teacher.backbone/epoch_294-step_491765.pth
    """
    return KaikoModel(model_str, out_indices)


@register_model("DINOv2_vitg14_from_imagenet_tcga_epoch_244")
def model(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = """
        # ckpt_path: az://experimental@stkaikodtpprdlab.blob.core.windows.net/pathology_fm/runs/nebul/vitg14_tcga_epoch_244-step_408415.pth
        ckpt_path: /mnt/vast01/shared/outputs/mikhail/runs/FM-18A8DB4F-B29F-474E-BFA1-C5E8ABD39986/teacher.backbone/epoch_244-step_408415.pth
        path: torch.hub.load
        arguments:
          repo_or_dir: facebookresearch/dinov2:main
          model: dinov2_vitg14
          pretrained: false
    """
    return KaikoModel(model_str, out_indices)


@register_model("DINOv2_vitg14_from_imagenet_tcga_epoch_244_concat")
def model(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = """
        path: backbones.Kaiko
        arguments:
          repo_or_dir: facebookresearch/dinov2:main
          model: dinov2_vitg14
          pretrained: false
          concat_mean_patch_tokens: true
          # ckpt_path: az://experimental@stkaikodtpprdlab.blob.core.windows.net/pathology_fm/runs/nebul/vitg14_tcga_epoch_244-step_408415.pth
          ckpt_path: /mnt/vast01/shared/outputs/mikhail/runs/FM-18A8DB4F-B29F-474E-BFA1-C5E8ABD39986/teacher.backbone/epoch_244-step_408415.pth
    """
    return KaikoModel(model_str, out_indices)


@register_model("DINOv2_vitb14_four_hsv0.4")
def model(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = """
    ckpt_path: /mnt/vast01/shared/experimental/pathology_fm/mikhail/runs/FM-9BD5863B-43F0-4C65-A005-9BA47E5BDFEE/lightning_logs/version_0/checkpoints/epoch_099-step_166700.ckpt
    ckpt_submodule: teacher.backbone
    path: torch.hub.load
    arguments:
      repo_or_dir: facebookresearch/dinov2:main
      model: dinov2_vitb14
      pretrained: false
    """
    return KaikoModel(model_str, out_indices)


@register_model("DINOv2_vitb14_four+TCGAFrozen")
def model(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = """
    ckpt_path: /mnt/vast01/shared/experimental/pathology_fm/mikhail/runs/FM-7988DC68-9306-4ADF-ACE4-73AB5FB3F257/lightning_logs/version_0/checkpoints/epoch_099-step_166700.ckpt
    ckpt_submodule: teacher.backbone
    path: torch.hub.load
    arguments:
      repo_or_dir: facebookresearch/dinov2:main
      model: dinov2_vitb14
      pretrained: false
    """
    return KaikoModel(model_str, out_indices)


@register_model("DINOv2_vitb14_four-GTEx")
def model(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = """
    ckpt_path: /mnt/vast01/shared/experimental/pathology_fm/mikhail/runs/FM-E21FA649-6231-4BC1-9E40-7DE1FE6DEBF6/lightning_logs/version_0/checkpoints/epoch_099-step_166700.ckpt
    ckpt_submodule: teacher.backbone
    path: torch.hub.load
    arguments:
      repo_or_dir: facebookresearch/dinov2:main
      model: dinov2_vitb14
      pretrained: false
    """
    return KaikoModel(model_str, out_indices)


@register_model("DINOv2_vitb14_four-NKI")
def model(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = """
    ckpt_path: /mnt/vast01/shared/experimental/pathology_fm/mikhail/runs/FM-45BAF471-5E99-4139-BCDB-552E4642EA0B/lightning_logs/version_0/checkpoints/epoch_099-step_166700.ckpt
    ckpt_submodule: teacher.backbone
    path: torch.hub.load
    arguments:
      repo_or_dir: facebookresearch/dinov2:main
      model: dinov2_vitb14
      pretrained: false
    """
    return KaikoModel(model_str, out_indices)


@register_model("DINOv2_vitb14_four-CPTAC")
def model(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = """
    ckpt_path: /mnt/vast01/shared/experimental/pathology_fm/mikhail/runs/FM-B54A1C09-EBE8-4927-AF11-9DE3ECE2DB1B/lightning_logs/version_0/checkpoints/epoch_099-step_166700.ckpt
    ckpt_submodule: teacher.backbone
    path: torch.hub.load
    arguments:
      repo_or_dir: facebookresearch/dinov2:main
      model: dinov2_vitb14
      pretrained: false
    """
    return KaikoModel(model_str, out_indices)


@register_model("DINOv2_vitb14_four_noHSV")
def model(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = """
    ckpt_path: /mnt/vast01/shared/experimental/pathology_fm/mikhail/runs/FM-A24D6478-3D04-4D09-B61A-3BBF97CCC503/lightning_logs/version_0/checkpoints/epoch_099-step_166700.ckpt
    ckpt_submodule: teacher.backbone
    path: torch.hub.load
    arguments:
      repo_or_dir: facebookresearch/dinov2:main
      model: dinov2_vitb14
      pretrained: false
    """
    return KaikoModel(model_str, out_indices)


@register_model("DINOv2_vitb14_four_hsv0.2")
def model(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = """
    ckpt_path: /mnt/vast01/shared/experimental/pathology_fm/mikhail/runs/FM-EE5084B1-88CC-472F-B080-ADBE19FA34E9/lightning_logs/version_0/checkpoints/epoch_099-step_166700.ckpt
    ckpt_submodule: teacher.backbone
    path: torch.hub.load
    arguments:
      repo_or_dir: facebookresearch/dinov2:main
      model: dinov2_vitb14
      pretrained: false
    """
    return KaikoModel(model_str, out_indices)


@register_model("DINOv2_vitb14_four_hsv0.6-20")
def model(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = """
    ckpt_path: /mnt/vast01/shared/experimental/pathology_fm/mikhail/runs/FM-2D897A1E-F43F-4E66-920B-A686B59C0C35/lightning_logs/version_0/checkpoints/epoch_099-step_166700.ckpt
    ckpt_submodule: teacher.backbone
    path: torch.hub.load
    arguments:
      repo_or_dir: facebookresearch/dinov2:main
      model: dinov2_vitb14
      pretrained: false
    """
    return KaikoModel(model_str, out_indices)


@register_model("DINOv2_vitb14_four_hsv0.6-10")
def model(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = """
    ckpt_path: /mnt/vast01/shared/experimental/pathology_fm/mikhail/runs/FM-1B3C1C64-E641-4333-B129-A5984BBDBA84/lightning_logs/version_0/checkpoints/epoch_099-step_166700.ckpt
    ckpt_submodule: teacher.backbone
    path: torch.hub.load
    arguments:
      repo_or_dir: facebookresearch/dinov2:main
      model: dinov2_vitb14
      pretrained: false
    """
    return KaikoModel(model_str, out_indices)


@register_model("DINOv2_vitb14_four_hsv0.4+TCGAfrozen")
def model(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = """
    ckpt_path: /mnt/vast01/shared/experimental/pathology_fm/mikhail/runs/6F8BB4BA-BCC4-47DF-B11D-65E0E2E95156/lightning_logs/version_0/checkpoints/epoch_099-step_166700.ckpt
    ckpt_submodule: teacher.backbone
    path: torch.hub.load
    arguments:
      repo_or_dir: facebookresearch/dinov2:main
      model: dinov2_vitb14
      pretrained: false
    """
    return KaikoModel(model_str, out_indices)


@register_model("DINOv2_vitb14_four_mf0.2_hsv0.4")
def model(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = """
    ckpt_path: /mnt/vast01/shared/experimental/pathology_fm/mikhail/runs/7C85A1BD-BA78-4BEE-A7EB-2DC3F2F1309A/lightning_logs/version_0/checkpoints/epoch_099-step_166700.ckpt
    ckpt_submodule: teacher.backbone
    path: torch.hub.load
    arguments:
      repo_or_dir: facebookresearch/dinov2:main
      model: dinov2_vitb14
      pretrained: false
    """
    return KaikoModel(model_str, out_indices)


@register_model("DINOv2_vitb14_four-GTEx_mf0.2_hsv0.4")
def model(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = """
    ckpt_path: /mnt/vast01/shared/experimental/pathology_fm/mikhail/runs/D5B72462-D6B2-4E20-B6AC-42B756642639/lightning_logs/version_0/checkpoints/epoch_099-step_166700.ckpt
    ckpt_submodule: teacher.backbone
    path: torch.hub.load
    arguments:
      repo_or_dir: facebookresearch/dinov2:main
      model: dinov2_vitb14
      pretrained: false
    """
    return KaikoModel(model_str, out_indices)


@register_model("DINOv2_vitb14_four-GTEx+TCGAFrozen_mf0.2_hsv0.4")
def model(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = """
    ckpt_path: /mnt/vast01/shared/experimental/pathology_fm/mikhail/runs/A45721C7-F3C2-46DC-9B41-1B271A8D7DCF/lightning_logs/version_0/checkpoints/epoch_099-step_166700.ckpt
    ckpt_submodule: teacher.backbone
    path: torch.hub.load
    arguments:
      repo_or_dir: facebookresearch/dinov2:main
      model: dinov2_vitb14
      pretrained: false
    """
    return KaikoModel(model_str, out_indices)


@register_model("DINOv2_vitg14_nki-tcga_post_100_aspect_epoch_059_resize392")
def model(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = """
    path: backbones.Kaiko
    arguments:
      repo_or_dir: facebookresearch/dinov2:main
      model: dinov2_vitg14
      pretrained: false
      concat_mean_patch_tokens: false
      resize: 392
      ckpt_path: /mnt/vast01/shared/experimental/pathology_fm/mikhail/runs/E48D4DAD-C48E-49E5-B4D9-AB679A594090/lightning_logs/version_0/checkpoints/epoch_059-step_30000.ckpt
      ckpt_submodule: teacher.backbone
      mode: bicubic
    """
    return KaikoModel(model_str, out_indices)


@register_model("DINOv2_vitg14_nki-tcga_post_100_aspect_epoch_059_bicubic_concat_resize392")
def model(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = """
    path: backbones.Kaiko
    arguments:
      repo_or_dir: facebookresearch/dinov2:main
      model: dinov2_vitg14
      pretrained: false
      concat_mean_patch_tokens: true
      resize: 392
      ckpt_path: /mnt/vast01/shared/experimental/pathology_fm/mikhail/runs/E48D4DAD-C48E-49E5-B4D9-AB679A594090/lightning_logs/version_0/checkpoints/epoch_059-step_30000.ckpt
      ckpt_submodule: teacher.backbone
      mode: bicubic
    """
    return KaikoModel(model_str, out_indices)


@register_model("KAIKO-vitB8")
def model(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    return torch.hub.load(  # type: ignore
        repo_or_dir="kaiko-ai/towards_large_pathology_fms",
        model="vitb8",
        trust_repo=True,
        dynamic_img_size=dynamic_img_size,
        out_indices=out_indices,
    )


@register_model("KAIKO-vitB8_concat")
def model(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = f"""
    path: backbones.Kaiko
    arguments:
      repo_or_dir: kaiko-ai/towards_large_pathology_fms
      model: vitb8
      concat_mean_patch_tokens: true
      trust_repo: true
      dynamic_img_size: {dynamic_img_size}
      out_indices: {"null" if out_indices is None else out_indices}
    """
    return KaikoModel(model_str, None)


@register_model("dino_vitL16_phikon2")
def model(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = f"""
    path: renormalized.RenormalizingModel
    arguments:
      new_normalization:
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
      model:
        path: backbones.HuggingFaceModel
        arguments:
          model_name_or_path: owkin/phikon-v2
          output_transform:
            class_path: extract_cls_token.{"ExtractCLSToken" if out_indices is None else "ExtractPatchFeatures"}
    """
    return KaikoModel(model_str, None)


@register_model("dino_vits16_phikon")
def model(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = f"""
    path: renormalized.RenormalizingModel
    arguments:
      new_normalization:
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
      model:
        path: backbones.HuggingFaceModel
        arguments:
          model_name_or_path: owkin/phikon
          output_transform:
            class_path: extract_cls_token.{"ExtractCLSToken" if out_indices is None else "ExtractPatchFeatures"}
    """
    return KaikoModel(model_str, None)


@register_model("dino_vitL16_phikon2_concat")
def model(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = f"""
    path: renormalized.RenormalizingModel
    arguments:
      new_normalization:
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
      model:
        path: backbones.HuggingFaceModel
        arguments:
          model_name_or_path: owkin/phikon-v2
          output_transform:
            class_path: extract_cls_token.{"ExtractConcatToken" if out_indices is None else "ExtractPatchFeatures"}
    """
    return KaikoModel(model_str, None)


@register_model("vitg14_Kaiko_Midnight_concat")
def model(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = f"""
    path: backbones.HuggingFaceModel
    arguments:
      model_name_or_path: kaiko-ai/midnight
      output_transform:
        class_path: extract_cls_token.{"ExtractConcatToken" if out_indices is None else "ExtractPatchFeatures"}
    """
    return KaikoModel(model_str, None)


@register_model("dino_vits16_phikon_concat")
def model(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = f"""
    path: renormalized.RenormalizingModel
    arguments:
      new_normalization:
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
      model:
        path: backbones.HuggingFaceModel
        arguments:
          model_name_or_path: owkin/phikon
          output_transform:
            class_path: extract_cls_token.{"ExtractConcatToken" if out_indices is None else "ExtractPatchFeatures"}
    """
    return KaikoModel(model_str, None)


@register_model("vitL14_histai_hibou_l")
def model(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = f"""
    path: renormalized.RenormalizingModel
    arguments:
      new_normalization:
        mean: [0.7068,0.5755,0.722]
        std: [0.195,0.2316,0.1816]
      model:
        path: backbones.HuggingFaceModel
        arguments:
          model_name_or_path: histai/hibou-L
          trust_remote_code: true
          with_config: false
          output_transform:
            class_path: extract_cls_token.{"ExtractCLSToken" if out_indices is None else "ExtractPatchFeatures"}
            init_args: {'{}' if out_indices is None else '{num_reg_tokens: 4}'}
    """
    return KaikoModel(model_str, None)


@register_model("vitL14_histai_hibou_l_concat")
def model(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = f"""
    path: renormalized.RenormalizingModel
    arguments:
      new_normalization:
        mean: [0.7068,0.5755,0.722]
        std: [0.195,0.2316,0.1816]
      model:
        path: backbones.HuggingFaceModel
        arguments:
          model_name_or_path: histai/hibou-L
          trust_remote_code: true
          with_config: false
          output_transform:
            class_path: extract_cls_token.{"ExtractConcatToken" if out_indices is None else "ExtractPatchFeatures"}
            init_args:
              num_reg_tokens: 4
    """
    return KaikoModel(model_str, None)


@register_model("vitg14_Prov_GigaPath")
def model(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = f"""
    path: renormalized.RenormalizingModel
    arguments:
      new_normalization:
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
      model:
        path: backbones.TimmModel
        arguments:
          concat_mean_patch_tokens: false
          out_indices: {"null" if out_indices is None else out_indices}
          features_only: {out_indices is not None}
          model_name: hf_hub:prov-gigapath/prov-gigapath
          pretrained: true
          dynamic_img_size: {dynamic_img_size}
    """
    return KaikoModel(model_str, None)


@register_model("vitg14_Prov_GigaPath_concat")
def model(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = f"""
    path: renormalized.RenormalizingModel
    arguments:
      new_normalization:
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
      model:
        path: backbones.TimmModel
        arguments:
          concat_mean_patch_tokens: true
          out_indices: {"null" if out_indices is None else out_indices}
          features_only: {out_indices is not None}
          model_name: hf_hub:prov-gigapath/prov-gigapath
          pretrained: true
          dynamic_img_size: {dynamic_img_size}
    """
    return KaikoModel(model_str, None)


@register_model("vits16_Lunit_renorm")
def model(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = f"""
    path: renormalized.RenormalizingModel
    arguments:
      new_normalization:
        mean: [0.70322989, 0.53606487, 0.66096631]
        std: [0.21716536, 0.26081574, 0.20723464]
      model:
        path: backbones.TimmModel
        arguments:
          concat_mean_patch_tokens: false
          out_indices: {"null" if out_indices is None else out_indices}
          features_only: {out_indices is not None}
          model_name: hf-hub:1aurent/vit_small_patch16_224.lunit_dino
          pretrained: true
          dynamic_img_size: true
          num_classes: 0
    """
    return KaikoModel(model_str, None)


@register_model("vits16_Lunit_renorm_concat")
def model(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = f"""
    path: renormalized.RenormalizingModel
    arguments:
      new_normalization:
        mean: [0.70322989, 0.53606487, 0.66096631]
        std: [0.21716536, 0.26081574, 0.20723464]
      model:
        path: backbones.TimmModel
        arguments:
          concat_mean_patch_tokens: true
          out_indices: {"null" if out_indices is None else out_indices}
          features_only: {out_indices is not None}
          model_name: hf-hub:1aurent/vit_small_patch16_224.lunit_dino
          pretrained: true
          dynamic_img_size: true
          num_classes: 0
    """
    return KaikoModel(model_str, None)


@register_model("vitL16_UNI")
def model(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = f"""
    path: renormalized.RenormalizingModel
    arguments:
      new_normalization:
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
      model:
        path: backbones.TimmModel
        arguments:
          concat_mean_patch_tokens: false
          out_indices: {"null" if out_indices is None else out_indices}
          features_only: {out_indices is not None}
          model_name: hf-hub:MahmoodLab/uni
          init_values: 1.0e-5
          pretrained: true
          dynamic_img_size: true
          num_classes: 0
    """
    return KaikoModel(model_str, None)


@register_model("vitL16_UNI_resize512")
def model(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = f"""
    path: renormalized.RenormalizingModel
    arguments:
      new_normalization:
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
      model:
        path: backbones.TimmModel
        arguments:
          concat_mean_patch_tokens: false
          out_indices: {"null" if out_indices is None else out_indices}
          features_only: {out_indices is not None}
          model_name: hf-hub:MahmoodLab/uni
          init_values: 1.0e-5
          pretrained: true
          dynamic_img_size: true
          num_classes: 0
    """
    return KaikoModel(model_str, None)


@register_model("vitL16_UNI_concat")
def model(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = f"""
    path: renormalized.RenormalizingModel
    arguments:
      new_normalization:
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
      model:
        path: backbones.TimmModel
        arguments:
          concat_mean_patch_tokens: true
          out_indices: {"null" if out_indices is None else out_indices}
          features_only: {out_indices is not None}
          model_name: hf-hub:MahmoodLab/uni
          init_values: 1.0e-5
          pretrained: true
          dynamic_img_size: true
          num_classes: 0
    """
    return KaikoModel(model_str, None)


@register_model("vitL16_UNI_concat_resize512")
def model(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = f"""
    path: renormalized.RenormalizingModel
    arguments:
      new_normalization:
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
      model:
        path: backbones.TimmModel
        arguments:
          concat_mean_patch_tokens: true
          out_indices: {"null" if out_indices is None else out_indices}
          features_only: {out_indices is not None}
          model_name: hf-hub:MahmoodLab/uni
          init_values: 1.0e-5
          pretrained: true
          dynamic_img_size: true
          num_classes: 0
    """
    return KaikoModel(model_str, None)


@register_model("vitg14_224_UNI2")
def model(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = f"""
    path: renormalized.RenormalizingModel
    arguments:
      new_normalization:
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
      model:
        path: backbones.TimmModel
        arguments:
          concat_mean_patch_tokens: false
          out_indices: {"null" if out_indices is None else out_indices}
          features_only: {out_indices is not None}
          model_name: hf-hub:MahmoodLab/UNI2-h
          pretrained: True
          img_size: 224
          patch_size: 14
          depth: 24
          num_heads: 24
          init_values: 1.0e-5
          embed_dim: 1536
          mlp_ratio: 5.33334  # 2.66667*2
          num_classes: 0
          no_embed_class: True
          mlp_layer: timm.layers.SwiGLUPacked
          act_layer: torch.nn.SiLU
          reg_tokens: 8
          dynamic_img_size: True
    """
    return KaikoModel(model_str, None)


@register_model("vitg14_224_UNI2_resize392")
def model(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = f"""
    path: renormalized.RenormalizingModel
    arguments:
      new_normalization:
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
      model:
        path: backbones.TimmModel
        arguments:
          concat_mean_patch_tokens: false
          out_indices: {"null" if out_indices is None else out_indices}
          features_only: {out_indices is not None}
          model_name: hf-hub:MahmoodLab/UNI2-h
          pretrained: True
          img_size: 224
          patch_size: 14
          depth: 24
          num_heads: 24
          init_values: 1.0e-5
          embed_dim: 1536
          mlp_ratio: 5.33334  # 2.66667*2
          num_classes: 0
          no_embed_class: True
          mlp_layer: timm.layers.SwiGLUPacked
          act_layer: torch.nn.SiLU
          reg_tokens: 8
          dynamic_img_size: True
    """
    return KaikoModel(model_str, None)


@register_model("vitg14_224_UNI2_concat")
def model(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = f"""
    path: renormalized.RenormalizingModel
    arguments:
      new_normalization:
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
      model:
        path: backbones.TimmModel
        arguments:
          concat_mean_patch_tokens: true
          out_indices: {"null" if out_indices is None else out_indices}
          features_only: {out_indices is not None}
          model_name: hf-hub:MahmoodLab/UNI2-h
          pretrained: True
          img_size: 224
          patch_size: 14
          depth: 24
          num_heads: 24
          init_values: 1.0e-5
          embed_dim: 1536
          mlp_ratio: 5.33334  # 2.66667*2
          num_classes: 0
          no_embed_class: True
          mlp_layer: timm.layers.SwiGLUPacked
          act_layer: torch.nn.SiLU
          reg_tokens: 8
          dynamic_img_size: True
    """
    return KaikoModel(model_str, None)


@register_model("vitg14_224_UNI2_concat_resize392")
def model(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = f"""
    path: renormalized.RenormalizingModel
    arguments:
      new_normalization:
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
      model:
        path: backbones.TimmModel
        arguments:
          concat_mean_patch_tokens: true
          out_indices: {"null" if out_indices is None else out_indices}
          features_only: {out_indices is not None}
          model_name: hf-hub:MahmoodLab/UNI2-h
          pretrained: True
          img_size: 224
          patch_size: 14
          depth: 24
          num_heads: 24
          init_values: 1.0e-5
          embed_dim: 1536
          mlp_ratio: 5.33334  # 2.66667*2
          num_classes: 0
          no_embed_class: True
          mlp_layer: timm.layers.SwiGLUPacked
          act_layer: torch.nn.SiLU
          reg_tokens: 8
          dynamic_img_size: True
    """
    return KaikoModel(model_str, None)


@register_model("vitH14_Virchow2")
def model(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = f"""
    path: renormalized.RenormalizingModel
    arguments:
      new_normalization:
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
      model:
        path: backbones.Virchow2
        arguments:
          concat_mean_patch_tokens: false
          out_indices: {"null" if out_indices is None else out_indices}
          features_only: {out_indices is not None}
    """
    return KaikoModel(model_str, None)


@register_model("vitH14_Virchow2_concat")
def model(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = f"""
    path: renormalized.RenormalizingModel
    arguments:
      new_normalization:
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
      model:
        path: backbones.Virchow2
        arguments:
          concat_mean_patch_tokens: true
          out_indices: {"null" if out_indices is None else out_indices}
          features_only: {out_indices is not None}
    """
    return KaikoModel(model_str, None)


@register_model("Bioptimus_h_optimus_0")
def model(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = f"""
    path: renormalized.RenormalizingModel
    arguments:
      new_normalization:
        mean: [0.707223, 0.578729, 0.703617]
        std: [0.211883, 0.230117, 0.177517]
      model:
        path: backbones.TimmModel
        arguments:
          concat_mean_patch_tokens: false
          model_name: hf-hub:bioptimus/H-optimus-0
          init_values: 1.0e-5
          pretrained: true
          dynamic_img_size: true
          num_classes: 0
          out_indices: {"null" if out_indices is None else out_indices}
          features_only: {out_indices is not None}
    """
    return KaikoModel(model_str, None)


@register_model("Bioptimus_h_optimus_0_concat")
def model(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    model_str = f"""
    path: renormalized.RenormalizingModel
    arguments:
      new_normalization:
        mean: [0.707223, 0.578729, 0.703617]
        std: [0.211883, 0.230117, 0.177517]
      model:
        path: backbones.TimmModel
        arguments:
          concat_mean_patch_tokens: true
          model_name: hf-hub:bioptimus/H-optimus-0
          init_values: 1.0e-5
          pretrained: true
          dynamic_img_size: true
          num_classes: 0
          out_indices: {"null" if out_indices is None else out_indices}
          features_only: {out_indices is not None}
    """
    return KaikoModel(model_str, None)
