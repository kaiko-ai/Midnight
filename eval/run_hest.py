#!/usr/bin/env python3
"""
Run HEST benchmark on a custom ViT checkpoint.
"""

import argparse
import os
import socket
import sys
from pathlib import Path

import torch
import torchvision.transforms as T
import yaml
from hest.bench import benchmark
from loguru import logger
from torchvision.transforms import v2

# External libraries
from object_tools import ModelBuilder

# Set the default timeout (in seconds)
socket.setdefaulttimeout(50)


def main():
    parser = argparse.ArgumentParser(
        description="Run HEST benchmark on a ViT-S/14 DINOv2 model with distilled weights."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./hest_bench_config.yaml",
        help="Path to the HEST benchmark config YAML file.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="./vits14_distilled_from_tcga-nki_099_test.pth",
        help="Path to the distilled model checkpoint file.",
    )
    parser.add_argument(
        "--repo-or-dir",
        type=str,
        default="facebookresearch/dinov2:main",
        help="mlfhub repository or local directory for the base model.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="dinov2_vits14",
        help="Name of the model to load from the repository.",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Load base model with pretrained weights (default: False).",
    )
    args = parser.parse_args()

    # Initialize logging
    logger.remove()  # Remove any default handlers to reconfigure
    logger.add(sys.stderr, level="INFO", format="<green>{time}</green> | <level>{message}</level>")

    # Resolve paths
    config_path = Path(args.config).resolve()

    # Validate config exists
    if not config_path.is_file():
        logger.error(f"Config file not found at: {config_path}")
        sys.exit(1)

    # Build transforms
    model_transforms = v2.Compose(
        [
            v2.Resize(size=224),  # FYI: change to 392 for the Midnight-92k/392 models
            # v2.CenterCrop(size=224),
            T.ToTensor(),
            T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )

    # Load model
    with open(config_path) as f:
        config = yaml.safe_load(f)

    custom_encoders = config["custom_encoders"]
    config.pop("custom_encoders")
    for name, model_args in custom_encoders.items():
        model = ModelBuilder(**model_args).build()

        config["results_dir"] += "/" + name
        config["embed_dataroot"] += "/" + name
        config["weights_root"] += "/" + name
        model_config_path = config["results_dir"] + "/config.yaml"
        os.mkdir(config["results_dir"])
        with open(model_config_path, "w") as yaml_file:
            yaml.dump(config, yaml_file)

        # Run benchmark
        try:
            logger.info(f"Running HEST benchmark for model: {name}")
            benchmark(
                model,
                model_transforms,
                precision=torch.float32,
                config=model_config_path,
            )
            logger.info("Benchmark completed successfully.")
        except Exception as e:
            logger.exception(f"Benchmark failed: {e}")


if __name__ == "__main__":
    main()
