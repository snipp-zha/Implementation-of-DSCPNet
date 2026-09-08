import argparse
import os

import torch

from config import Config, update_config_from_args
from data import build_datasets
from engine import evaluate, set_seed, train_one_epoch
from model import DSCPNet


def parse_args():
    parser = argparse.ArgumentParser(description="Train DSCPNet on base classes.")
    parser.add_argument("--data_root", type=str)
    parser.add_argument("--shot", type=int)
    parser.add_argument("--query", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--train_episodes", type=int)
    parser.add_argument("--test_episodes", type=int)
    parser.add_argument("--checkpoint", type=str)
    return parser.parse_args()


def main():
    cfg = update_config_from_args(Config(), parse_args())
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    train_set, test_set = build_datasets(cfg)
    model = DSCPNet(
        num_base_classes=cfg.num_base_classes,
        temperature=cfg.temperature,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    best_acc = 0.0
    os.makedirs(os.path.dirname(cfg.checkpoint) or ".", exist_ok=True)
    for epoch in range(1, cfg.epochs + 1):
        loss, train_acc = train_one_epoch(model, train_set, optimizer, cfg, device)
        test_acc, test_std, test_ci = evaluate(model, test_set, cfg, device)
        print(
            "Epoch {:03d}: loss={:.4f}, train_acc={:.2f}, "
            "test_acc={:.2f}, std={:.2f}, ci95={:.2f}".format(
                epoch, loss, train_acc, test_acc, test_std, test_ci
            )
        )
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({"model": model.state_dict(), "cfg": vars(cfg)}, cfg.checkpoint)
            print("Saved checkpoint to {}".format(cfg.checkpoint))


if __name__ == "__main__":
    main()
