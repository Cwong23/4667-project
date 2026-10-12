import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

MODEL_DIR = Path(__file__).resolve().parent
DATA_DIR = MODEL_DIR / "data"
MANIFEST = DATA_DIR / "splits" / "split.json"

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # ImageNet stats

train_transform = T.Compose([
    T.RandomResizedCrop(224, scale=(0.6, 1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.2, 0.2, 0.2),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])
eval_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])


class FoodDataset(Dataset):
    def __init__(self, items, root=DATA_DIR, transform=eval_transform):
        self.items, self.root, self.transform = items, Path(root), transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        item = self.items[i]
        img = Image.open(self.root / item["path"]).convert("RGB")
        return self.transform(img), item["label"]


def load_manifest(path=MANIFEST):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def stratified_val_split(items, val_frac, seed):
    by_label = defaultdict(list)
    for it in items:
        by_label[it["label"]].append(it)
    rng = random.Random(seed)
    train, val = [], []
    for group in by_label.values():
        rng.shuffle(group)
        n_val = max(1, round(len(group) * val_frac))
        val += group[:n_val]
        train += group[n_val:]
    return train, val


def make_dataloaders(batch_size=32, val_frac=0.1, workers=4, seed=42):
    manifest = load_manifest()
    train_items, val_items = stratified_val_split(
        manifest["train"], val_frac, seed)
    pin = torch.cuda.is_available()

    def loader(items, tf, shuffle):
        return DataLoader(
            FoodDataset(items, DATA_DIR, tf), batch_size=batch_size, shuffle=shuffle,
            num_workers=workers, pin_memory=pin, persistent_workers=workers > 0,
        )

    return (
        loader(train_items, train_transform, True),
        loader(val_items, eval_transform, False),
        loader(manifest["test"], eval_transform, False),
        manifest["classes"],
    )
