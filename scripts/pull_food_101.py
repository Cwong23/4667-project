import json
from pathlib import Path

from torchvision.datasets import Food101

SCRIPTS_DIR = Path(__file__).resolve().parent
DATA_DIR = (SCRIPTS_DIR / ".." / "model" / "data").resolve()
LABELS_JSON = SCRIPTS_DIR / "labels.json"


def split_labels(ds, dataset_root):
    return {
        f.relative_to(dataset_root).as_posix(): label
        for f, label in zip(ds._image_files, ds._labels)
    }


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    train = Food101(root=str(DATA_DIR), split="train", download=True)
    test = Food101(root=str(DATA_DIR), split="test", download=True)

    dataset_root = DATA_DIR / "food-101"

    out = {
        "classes": train.classes,
        "class_to_idx": train.class_to_idx,
        "train": split_labels(train, dataset_root),
        "test": split_labels(test, dataset_root),
    }

    with open(LABELS_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Dataset: {dataset_root}")
    print(f"Labels:  {LABELS_JSON}")
    print(
        f"{len(out['classes'])} classes, {len(out['train'])} train, {len(out['test'])} test")


if __name__ == "__main__":
    main()
