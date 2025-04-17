"""Python object related utilities and helpers."""

import copy
import dataclasses
import importlib
from typing import Any, Union

import torch
from jsonargparse import ArgumentParser
from lightning.fabric.utilities.cloud_io import _load as pl_load
from lightning.fabric.utilities.cloud_io import get_filesystem
from loguru import logger

SERIALIZABLE_TYPES = Union[int, float, str, bool, None]  # noqa: UP007
SERIALIZABLE_TYPES_COMPOSITE = Union[  # noqa: UP007
    SERIALIZABLE_TYPES,
    list[Union[SERIALIZABLE_TYPES, "SERIALIZABLE_TYPES_COMPOSITE"]],
    dict[str, Union[SERIALIZABLE_TYPES, "SERIALIZABLE_TYPES_COMPOSITE"]],
]


@dataclasses.dataclass
class ObjectBuilder:
    """Helper dataclass which allows to initialize objects on command."""

    path: str
    """The object path (class or function)."""

    arguments: dict[str, SERIALIZABLE_TYPES_COMPOSITE] | None = None
    """The initialization arguments of the object."""

    def build(self) -> Any:
        """Initializes and returns the defined object."""
        return _build_object_from_path(
            self.path,
            get_anyobject_jsonargparse(copy.deepcopy(self.arguments)) if self.arguments else None,
        )

    def to_dict(self) -> dict[str, Any]:
        """Converts the object builder to a dictionary."""
        return dataclasses.asdict(self)


@dataclasses.dataclass
class ModelBuilder(ObjectBuilder):
    """Helper dataclass for model initialization."""

    ckpt_path: str | None = None
    ckpt_submodule: str | None = None

    def build(self) -> Any:
        """Initializes and returns the defined model."""
        model = super().build()
        if self.ckpt_path is not None:
            model = load_model_checkpoint(model, self.ckpt_path, ckpt_submodule=self.ckpt_submodule)
        return model


def _build_object_from_path(path: str, arguments: dict[str, Any] | None) -> Any:
    """Initializes and build an object from path.

    Args:
        path: The path to the object (class or function).
        arguments: The initialization arguments. Defaults to `None`.

    Returns:
        The path object.
    """
    module_name, class_name = path.rsplit(".", 1)
    try:
        _module = importlib.import_module(module_name)
        try:
            _object = getattr(_module, class_name)(**arguments or {})
        except AttributeError as err:
            raise AttributeError(
                f"Class `{class_name}` in `{module_name}` does not exist."
            ) from err
    except ImportError as err:
        raise ImportError(f"Module `{module_name}` does not exist.") from err
    return _object


def get_anyobject_jsonargparse(conf_dict: dict[str, Any], expected_type=Any) -> Any:
    """Use jsonargparse to parse arbitrary object."""
    parser = ArgumentParser()
    parser.add_argument("arg", type=expected_type)
    anyobject = parser.parse_object({"arg": conf_dict})
    return parser.instantiate_classes(anyobject).arg


def load_model_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    strict: bool = False,
    ckpt_submodule: str | None = None,
) -> torch.nn.Module:
    """Initializes the model with the weights.

    Args:
        model: model to initialize.
        checkpoint_path: the path to the checkpoint.
        strict: if `True`, it loads the weights only if the dictionary matches the architecture
            exactly. if `False`, it loads the weights even if the weights of some layers
            are missing.
        ckpt_submodule: the submodule of the checkpoint for loading into the model. If `None`, load
            the entire checkpoint. Default: `None`.

    Returns:
        the model initialized with the checkpoint.
    """
    logger.info(f"Loading {model.__class__.__name__} from checkpoint {checkpoint_path}")
    fs = get_filesystem(checkpoint_path, anon=False)
    with fs.open(checkpoint_path, "rb") as f:
        checkpoint = pl_load(f, map_location="cpu")  # type: ignore[arg-type]
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        if ckpt_submodule is not None:
            key = ckpt_submodule if ckpt_submodule.endswith(".") else ckpt_submodule + "."
            checkpoint = {
                m.removeprefix(key): w for m, w in checkpoint.items() if m.startswith(key)
            }
        out = model.load_state_dict(checkpoint, strict=strict)
        missing, unexpected = out.missing_keys, out.unexpected_keys
        keys = model.state_dict().keys()
        if len(missing):
            logger.warning(
                f"{len(missing)}/{len(keys)} modules are missing in the checkpoint and will not be "
                f"initialized: {missing}"
            )
        if len(unexpected):
            logger.warning(
                f"The checkpoint also contains {len(unexpected)} modules ignored by the model: "
                f"{unexpected}"
            )
        logger.info(
            f"Loaded {len(set(keys) - set(missing))}/{len(keys)} modules for "
            f"{model.__class__.__name__} from checkpoint {checkpoint_path}"
        )
    return model
