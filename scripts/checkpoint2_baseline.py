"""Checkpoint 2 baseline: name the dish in a photo.

Checkpoint 1 treated calories as a regression target of food name and serving
size. That table has one row per dish, so it cannot look at a photo. This
script trains the missing piece: a 27-way classifier on the Food-101 dishes
that have a calorie-table entry.

The trained model is L2 logistic regression on a frozen ImageNet ResNet18
embedding. The backbone is feature extraction, not a model we fit. Test images
come from Food-101's official test split and are never used to fit the
classifier or the normalization constants.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import Food101
from torchvision.models import ResNet18_Weights, resnet18

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "model" / "data"
CALORIES_PATH = ROOT / "scripts" / "food_calories.json"
OUT_DIR = ROOT / "results" / "checkpoint2"
CACHE_PATH = DATA_DIR / "checkpoint2_features.npz"

BATCH_SIZE = 64
RANDOM_STATE = 42


class RemappedFood(Dataset):
    """Food-101 examples whose labels are remapped onto the calorie classes."""

    def __init__(self, base: Food101, indices: list[int], remap: dict[int, int]):
        self.base = base
        self.indices = indices
        self.remap = remap

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        image, label = self.base[self.indices[i]]
        return image, self.remap[int(label)]


def load_calorie_table() -> tuple[list[str], list[str]]:
    rows = json.loads(CALORIES_PATH.read_text(encoding="utf-8"))
    classes = [row["food_item"] for row in rows]
    if len(classes) != len(set(classes)):
        raise SystemExit("food_calories.json has duplicate food_item names")
    missing_grams = [row["food_item"] for row in rows if row["grams"] is None]
    missing_calories = [row["food_item"] for row in rows if row["calories"] is None]
    if missing_calories:
        raise SystemExit(f"calorie table is missing calories for {missing_calories}")
    return classes, missing_grams


def subset_indices(dataset: Food101, classes: list[str]) -> tuple[list[int], dict[int, int]]:
    keep = set(classes)
    unknown = sorted(keep - set(dataset.classes))
    if unknown:
        raise SystemExit(f"calorie classes missing from Food-101: {unknown}")
    remap = {dataset.classes.index(name): new_i for new_i, name in enumerate(classes)}
    indices = [i for i, label in enumerate(dataset._labels) if int(label) in remap]
    return indices, remap


def extract_features(
    dataset: Food101,
    indices: list[int],
    remap: dict[int, int],
    backbone: torch.nn.Module,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(
        RemappedFood(dataset, indices, remap),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    features, labels = [], []
    backbone.eval()
    with torch.no_grad():
        for batch_i, (images, batch_labels) in enumerate(loader, start=1):
            embedding = backbone(images.to(device))
            features.append(embedding.cpu().numpy())
            labels.append(batch_labels.numpy())
            if batch_i == 1 or batch_i % 25 == 0 or batch_i == len(loader):
                print(f"  batch {batch_i}/{len(loader)}", flush=True)
    return np.concatenate(features).astype(np.float32), np.concatenate(labels)


def fit_baseline(x_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    # C is the inverse of L2 strength. l1_ratio=0 keeps the penalty pure L2.
    # max_iter is raised above the default because 27-way softmax on 512
    # features does not always converge in 100 lbfgs steps.
    clf = LogisticRegression(
        C=1.0,
        l1_ratio=0.0,
        solver="lbfgs",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    clf.fit(x_train, y_train)
    return clf


def top_confusions(cm: np.ndarray, classes: list[str], n: int = 8) -> list[dict]:
    pairs = []
    for true_i in range(cm.shape[0]):
        for pred_i in range(cm.shape[1]):
            if true_i == pred_i:
                continue
            count = int(cm[true_i, pred_i])
            if count:
                pairs.append(
                    {
                        "true": classes[true_i],
                        "predicted": classes[pred_i],
                        "count": count,
                    }
                )
    pairs.sort(key=lambda row: row["count"], reverse=True)
    return pairs[:n]


def save_plots(
    cm: np.ndarray,
    classes: list[str],
    y_test: np.ndarray,
    y_pred: np.ndarray,
    test_paths: list[str],
    proba: np.ndarray,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    recall = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(recall, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=90, fontsize=8)
    ax.set_yticks(range(len(classes)), classes, fontsize=8)
    ax.set_xlabel("Predicted dish")
    ax.set_ylabel("True dish")
    ax.set_title("Test-set confusion matrix (row-normalized)")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Share of true class")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    per_class_recall = recall.diagonal()
    order = np.argsort(per_class_recall)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(np.array(classes)[order], per_class_recall[order], color="#3b6ea5")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Recall on the official test split")
    ax.set_title("Which dishes the baseline recognizes")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "per_class_recall.png", dpi=150)
    plt.close(fig)

    confidence = proba[np.arange(len(y_pred)), y_pred]
    wrong = np.flatnonzero(y_pred != y_test)
    show = wrong[np.argsort(-confidence[wrong])][:8]
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for ax, index in zip(axes.ravel(), show):
        ax.imshow(Image.open(test_paths[index]).convert("RGB"))
        ax.set_title(
            f"{classes[y_test[index]]}\n→ {classes[y_pred[index]]}",
            fontsize=8,
        )
        ax.axis("off")
    fig.suptitle("High-confidence test mistakes")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "high_confidence_mistakes.png", dpi=150)
    plt.close(fig)


def main() -> None:
    classes, missing_grams = load_calorie_table()
    print(f"{len(classes)} calorie-matched classes")
    print(f"missing serving grams: {missing_grams or 'none'}")

    weights = ResNet18_Weights.DEFAULT
    transform = weights.transforms()
    print(f"image transform: {transform}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"feature device: {device}")

    train_ds = Food101(str(DATA_DIR), split="train", transform=transform, download=True)
    test_ds = Food101(str(DATA_DIR), split="test", transform=transform, download=True)
    train_idx, remap = subset_indices(train_ds, classes)
    test_idx, _ = subset_indices(test_ds, classes)
    print(f"official split kept: train {len(train_idx)} | test {len(test_idx)}")

    if CACHE_PATH.exists():
        print(f"loading cached features from {CACHE_PATH}")
        cached = np.load(CACHE_PATH)
        x_train, y_train = cached["x_train"], cached["y_train"]
        x_test, y_test = cached["x_test"], cached["y_test"]
        if list(cached["classes"]) != classes:
            raise SystemExit("feature cache was built for a different class list")
    else:
        backbone = resnet18(weights=weights)
        backbone.fc = torch.nn.Identity()
        backbone = backbone.to(device).eval()
        print("extracting train features")
        x_train, y_train = extract_features(train_ds, train_idx, remap, backbone, device)
        print("extracting test features")
        x_test, y_test = extract_features(test_ds, test_idx, remap, backbone, device)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            CACHE_PATH,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            classes=np.array(classes),
        )

    test_paths = [str(test_ds._image_files[i]) for i in test_idx]
    print(f"X_train.shape {tuple(x_train.shape)}  y_train.shape {tuple(y_train.shape)}")
    print(f"X_test.shape  {tuple(x_test.shape)}  y_test.shape  {tuple(y_test.shape)}")
    sample_image, _ = train_ds[train_idx[0]]
    print(f"one image tensor going into the backbone: {tuple(sample_image.shape)}")

    clf = fit_baseline(x_train, y_train)
    train_pred = clf.predict(x_train)
    test_pred = clf.predict(x_test)
    test_proba = clf.predict_proba(x_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    test_macro_f1 = f1_score(y_test, test_pred, average="macro")
    majority = int(np.bincount(y_train).argmax())
    chance_acc = float(np.mean(y_test == majority))
    cm = confusion_matrix(y_test, test_pred, labels=np.arange(len(classes)))
    report = classification_report(
        y_test, test_pred, target_names=classes, digits=3, zero_division=0
    )
    confusions = top_confusions(cm, classes)

    print(f"majority-class accuracy {chance_acc:.3f}")
    print(f"train accuracy {train_acc:.3f}")
    print(f"test accuracy  {test_acc:.3f}")
    print(f"test macro F1  {test_macro_f1:.3f}")
    print(report)
    print("top confusions:", confusions)

    save_plots(cm, classes, y_test, test_pred, test_paths, test_proba)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, test_pred, labels=np.arange(len(classes)), zero_division=0
    )
    per_class = [
        {
            "dish": name,
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, name in enumerate(classes)
    ]

    summary = {
        "classes": classes,
        "missing_grams": missing_grams,
        "image_shape": list(sample_image.shape),
        "x_train_shape": list(x_train.shape),
        "y_train_shape": list(y_train.shape),
        "x_test_shape": list(x_test.shape),
        "y_test_shape": list(y_test.shape),
        "train_per_class": int(np.bincount(y_train).min()),
        "test_per_class": int(np.bincount(y_test).min()),
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "test_macro_f1": test_macro_f1,
        "majority_class_accuracy": chance_acc,
        "top_confusions": confusions,
        "per_class": per_class,
        "hyperparameters": {
            "backbone": "resnet18",
            "backbone_weights": "ImageNet-1K DEFAULT",
            "embedding_dim": int(x_train.shape[1]),
            "classifier": "logistic regression",
            "C": 1.0,
            "l1_ratio": 0.0,
            "solver": "lbfgs",
            "max_iter": 1000,
            "random_state": RANDOM_STATE,
        },
        "n_iter": clf.n_iter_.tolist(),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT_DIR / "classification_report.txt").write_text(report, encoding="utf-8")
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
