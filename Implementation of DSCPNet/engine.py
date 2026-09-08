import math
import random

import numpy as np
import torch
import torch.nn.functional as F

from data import EpisodicDataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy(logits, labels):
    pred = torch.argmax(logits, dim=1)
    return (pred == labels).float().mean().item() * 100.0


def train_one_epoch(model, image_dataset, optimizer, cfg, device):
    model.train()
    episodic_dataset = EpisodicDataset(image_dataset, cfg.way, cfg.shot, cfg.query)
    total_loss, total_acc = 0.0, 0.0

    for _ in range(cfg.train_episodes):
        support_x, support_y, query_x, query_y = episodic_dataset.sample_episode()
        support_x = support_x.to(device)
        support_y = support_y.to(device)
        query_x = query_x.to(device)
        query_y = query_y.to(device)

        logits_metric = model.forward_episode(support_x, support_y, query_x, cfg.way)
        metric_loss = F.cross_entropy(logits_metric, query_y)

        all_x = torch.cat([support_x, query_x], dim=0)
        all_y = torch.cat([support_y, query_y], dim=0)
        logits_cls = model.forward_classification(all_x)
        cls_loss = F.cross_entropy(logits_cls, all_y)

        loss = cls_loss + cfg.lambda_metric * metric_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += accuracy(logits_metric.detach(), query_y)

    return total_loss / cfg.train_episodes, total_acc / cfg.train_episodes


@torch.no_grad()
def evaluate(model, image_dataset, cfg, device):
    model.eval()
    episodic_dataset = EpisodicDataset(image_dataset, cfg.way, cfg.shot, cfg.query)
    acc_list = []

    for _ in range(cfg.test_episodes):
        support_x, support_y, query_x, query_y = episodic_dataset.sample_episode()
        support_x = support_x.to(device)
        support_y = support_y.to(device)
        query_x = query_x.to(device)
        query_y = query_y.to(device)

        logits = model.forward_episode(support_x, support_y, query_x, cfg.way)
        acc_list.append(accuracy(logits, query_y))

    mean = float(np.mean(acc_list))
    std = float(np.std(acc_list, ddof=1))
    ci95 = 1.96 * std / math.sqrt(len(acc_list))
    return mean, std, ci95
