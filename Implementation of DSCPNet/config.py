from dataclasses import dataclass


@dataclass
class Config:
    data_root: str = "./data"
    train_split: str = "base"
    test_split: str = "novel"
    image_size: int = 224

    way: int = 5
    shot: int = 1
    query: int = 15
    train_episodes: int = 1000
    test_episodes: int = 600
    epochs: int = 50

    num_base_classes: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-3
    lambda_metric: float = 1.0
    temperature: float = 10.0

    seed: int = 2026
    num_workers: int = 4
    device: str = "cuda"
    checkpoint: str = "./dscpnet_resnet12.pth"


def update_config_from_args(cfg, args):
    for key, value in vars(args).items():
        if value is not None and hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg
