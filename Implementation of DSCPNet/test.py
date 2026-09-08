import argparse

import torch

from config import Config, update_config_from_args
from data import build_datasets
from engine import evaluate, set_seed
from model import DSCPNet


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DSCPNet on novel classes.")
    parser.add_argument("--data_root", type=str)
    parser.add_argument("--shot", type=int)
    parser.add_argument("--query", type=int)
    parser.add_argument("--test_episodes", type=int)
    parser.add_argument("--checkpoint", type=str)
    return parser.parse_args()


def main():
    cfg = update_config_from_args(Config(), parse_args())
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    _, test_set = build_datasets(cfg)
    model = DSCPNet(
        num_base_classes=cfg.num_base_classes,
        temperature=cfg.temperature,
    ).to(device)
    checkpoint = torch.load(cfg.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])

    mean, std, ci95 = evaluate(model, test_set, cfg, device)
    print("Accuracy: {:.2f} +/- {:.2f}".format(mean, std))
    print("95% CI: +/- {:.2f}".format(ci95))


if __name__ == "__main__":
    main()
