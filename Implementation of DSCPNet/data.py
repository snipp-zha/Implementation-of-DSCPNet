import os
import random
from typing import Dict, List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def build_transform(image_size: int, train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


class SportsImageFolder(Dataset):
    def __init__(self, root: str, transform=None):
        self.root = root
        self.transform = transform
        self.classes = sorted([
            name for name in os.listdir(root)
            if os.path.isdir(os.path.join(root, name))
        ])
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.samples = []
        self.class_to_images: Dict[int, List[str]] = {}

        for class_name in self.classes:
            class_idx = self.class_to_idx[class_name]
            class_dir = os.path.join(root, class_name)
            paths = [
                os.path.join(class_dir, name)
                for name in sorted(os.listdir(class_dir))
                if name.lower().endswith(IMG_EXTENSIONS)
            ]
            if not paths:
                continue
            self.class_to_images[class_idx] = paths
            self.samples.extend((path, class_idx) for path in paths)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def load_image(self, path: str):
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image


class EpisodicDataset:
    def __init__(self, dataset: SportsImageFolder, way: int, shot: int, query: int):
        self.dataset = dataset
        self.way = way
        self.shot = shot
        self.query = query
        self.valid_classes = [
            cls for cls, paths in dataset.class_to_images.items()
            if len(paths) >= shot + query
        ]
        if len(self.valid_classes) < way:
            raise ValueError("Not enough classes with sufficient samples.")

    def sample_episode(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        selected_classes = random.sample(self.valid_classes, self.way)
        support_images, query_images = [], []
        support_labels, query_labels = [], []

        for episode_label, class_idx in enumerate(selected_classes):
            paths = random.sample(
                self.dataset.class_to_images[class_idx],
                self.shot + self.query,
            )
            support_paths = paths[:self.shot]
            query_paths = paths[self.shot:]

            for path in support_paths:
                support_images.append(self.dataset.load_image(path))
                support_labels.append(episode_label)
            for path in query_paths:
                query_images.append(self.dataset.load_image(path))
                query_labels.append(episode_label)

        return (
            torch.stack(support_images, dim=0),
            torch.tensor(support_labels, dtype=torch.long),
            torch.stack(query_images, dim=0),
            torch.tensor(query_labels, dtype=torch.long),
        )


def build_datasets(cfg):
    train_root = os.path.join(cfg.data_root, cfg.train_split)
    test_root = os.path.join(cfg.data_root, cfg.test_split)
    train_set = SportsImageFolder(train_root, build_transform(cfg.image_size, train=True))
    test_set = SportsImageFolder(test_root, build_transform(cfg.image_size, train=False))
    return train_set, test_set
