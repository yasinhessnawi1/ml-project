"""
Configuration management utilities.

This module provides functions to load and manage configuration from YAML files,
handle command-line arguments, and set up the runtime environment.

Functions:
    load_config: Load configuration from YAML file
    save_config: Save configuration to YAML file
    merge_configs: Merge multiple configurations
    set_seed: Set random seeds for reproducibility
    setup_device: Configure and return compute device
"""

import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import yaml


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path (Union[str, Path]): Path to the YAML configuration file

    Returns:
        Dict[str, Any]: Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML is malformed

    Example:
        >>> config = load_config('config.yaml')
        >>> print(config['training']['batch_size'])
        32
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML config: {e}")

    return config


def save_config(
    config: Dict[str, Any], save_path: Union[str, Path]
) -> None:
    """
    Save configuration dictionary to a YAML file.

    Args:
        config (Dict[str, Any]): Configuration dictionary to save
        save_path (Union[str, Path]): Path where to save the YAML file

    Example:
        >>> config = {'training': {'batch_size': 32}}
        >>> save_config(config, 'experiments/configs/exp_001.yaml')
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def merge_configs(
    base_config: Dict[str, Any], override_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries, with override_config taking precedence.

    Performs deep merge - nested dictionaries are merged recursively.

    Args:
        base_config (Dict[str, Any]): Base configuration
        override_config (Dict[str, Any]): Configuration to override base with

    Returns:
        Dict[str, Any]: Merged configuration

    Example:
        >>> base = {'training': {'batch_size': 32, 'epochs': 50}}
        >>> override = {'training': {'batch_size': 64}}
        >>> merged = merge_configs(base, override)
        >>> merged['training']
        {'batch_size': 64, 'epochs': 50}
    """
    merged = base_config.copy()

    for key, value in override_config.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            # Recursive merge for nested dictionaries
            merged[key] = merge_configs(merged[key], value)
        else:
            # Override value
            merged[key] = value

    return merged


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility across all libraries.

    Sets seeds for:
    - Python random
    - NumPy
    - PyTorch (CPU and CUDA)
    - CUDNN (deterministic mode)

    Args:
        seed (int, optional): Random seed value. Defaults to 42.
        deterministic (bool, optional): If True, set CUDNN to deterministic mode.
            This may reduce performance but ensures reproducibility. Defaults to True.

    Example:
        >>> set_seed(42)  # All random operations will be deterministic
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # CUDNN
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Benchmark mode finds optimal algorithms (faster but non-deterministic)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    # Set environment variable for Python hashing
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_device(
    device_name: Optional[str] = None,
    cuda_visible_devices: Optional[str] = None,
) -> torch.device:
    """
    Set up and return the compute device (CPU, CUDA, or MPS for Apple Silicon).

    Args:
        device_name (Optional[str], optional): Device name ('cuda', 'cpu', 'mps').
            If None, automatically selects best available device. Defaults to None.
        cuda_visible_devices (Optional[str], optional): CUDA_VISIBLE_DEVICES value
            (e.g., '0' or '0,1'). Defaults to None.

    Returns:
        torch.device: PyTorch device object

    Example:
        >>> device = setup_device('cuda')
        >>> print(device)
        device(type='cuda', index=0)

        >>> device = setup_device()  # Auto-select
        >>> print(device)
        device(type='cuda', index=0)  # If GPU available, else CPU
    """
    # Set CUDA_VISIBLE_DEVICES if specified
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    # Auto-select device if not specified
    if device_name is None:
        if torch.cuda.is_available():
            device_name = "cuda"
        elif (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            device_name = "mps"  # Apple Silicon
        else:
            device_name = "cpu"

    # Create device
    device = torch.device(device_name)

    # Print device info
    if device.type == "cuda":
        print(f"Using device: {device}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
    else:
        print(f"Using device: {device}")

    return device


def get_config_value(
    config: Dict[str, Any], key_path: str, default: Any = None
) -> Any:
    """
    Get a nested configuration value using dot notation.

    Args:
        config (Dict[str, Any]): Configuration dictionary
        key_path (str): Dot-separated path to the value (e.g., 'training.batch_size')
        default (Any, optional): Default value if key not found. Defaults to None.

    Returns:
        Any: Configuration value or default

    Example:
        >>> config = {'training': {'optimizer': {'lr': 0.001}}}
        >>> lr = get_config_value(config, 'training.optimizer.lr')
        >>> print(lr)
        0.001

        >>> missing = get_config_value(config, 'training.missing_key', default=42)
        >>> print(missing)
        42
    """
    keys = key_path.split(".")
    value = config

    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration to ensure all required fields are present.

    Args:
        config (Dict[str, Any]): Configuration dictionary to validate

    Raises:
        ValueError: If required configuration fields are missing or invalid

    Example:
        >>> config = load_config('config.yaml')
        >>> validate_config(config)  # Raises ValueError if invalid
    """
    required_keys = [
        "project",
        "paths",
        "dataset",
        "preprocessing",
        "training",
    ]

    for key in required_keys:
        if key not in config:
            raise ValueError(
                f"Missing required configuration section: {key}"
            )

    # Validate paths exist or can be created
    paths = config.get("paths", {})
    for path_key, path_value in paths.items():
        path = Path(path_value)
        if not path.exists() and path_key not in [
            "data_dir"
        ]:  # data_dir may not exist yet
            try:
                path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise ValueError(f"Cannot create path {path}: {e}")

    # Validate numeric values
    training = config.get("training", {})
    if training.get("batch_size", 0) <= 0:
        raise ValueError("batch_size must be positive")
    if training.get("num_epochs", 0) <= 0:
        raise ValueError("num_epochs must be positive")

    # Validate splits sum to 1.0
    dataset = config.get("dataset", {})
    splits_sum = sum(
        [
            dataset.get("train_split", 0),
            dataset.get("val_split", 0),
            dataset.get("test_split", 0),
        ]
    )
    if not (0.99 < splits_sum < 1.01):  # Allow small floating point error
        raise ValueError(
            f"Dataset splits must sum to 1.0, got {splits_sum}"
        )


def create_experiment_dir(
    base_dir: Union[str, Path], experiment_name: str
) -> Path:
    """
    Create a directory for experiment outputs.

    Creates directory structure:
    - {base_dir}/{experiment_name}/
        - checkpoints/
        - logs/
        - results/
        - configs/

    Args:
        base_dir (Union[str, Path]): Base directory for experiments
        experiment_name (str): Name of the experiment

    Returns:
        Path: Path to the created experiment directory

    Example:
        >>> exp_dir = create_experiment_dir('experiments', 'exp_001')
        >>> print(exp_dir)
        experiments/exp_001
    """
    exp_dir = Path(base_dir) / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (exp_dir / "checkpoints").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    (exp_dir / "results").mkdir(exist_ok=True)
    (exp_dir / "configs").mkdir(exist_ok=True)

    return exp_dir


# Example usage
if __name__ == "__main__":
    # Load configuration
    config = load_config("config.yaml")

    # Set random seed
    set_seed(config["project"]["seed"])

    # Setup device
    device = setup_device(config["project"]["device"])

    # Validate config
    validate_config(config)

    print("Configuration loaded and validated successfully!")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Number of epochs: {config['training']['num_epochs']}")
