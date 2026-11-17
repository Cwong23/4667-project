"""
Usage:
    python scripts/predict.py path/to/image.jpg
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import joblib
import numpy as np
import torch
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "model" / "checkpoints" / "checkpoint2_logreg.joblib"
CALORIES_PATH = ROOT / "scripts" / "food_calories.json"


def load_backbone(device: torch.device):
    """Same frozen feature extractor and transform as checkpoint2_baseline.py."""
    weights = ResNet18_Weights.DEFAULT
    backbone = resnet18(weights=weights)
    backbone.fc = torch.nn.Identity()
    return backbone.to(device).eval(), weights.transforms()


@torch.no_grad()
def embed(image_path: Path, backbone, transform, device) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)
    return backbone(x).cpu().numpy().astype(np.float32)


def describe_calories(entry: dict | None, grams: float | None) -> str:
    if entry is None:
        return "no calorie data"
    cal, serving = entry["calories"], entry["grams"]
    if grams is not None:
        if not serving:
            return f"{cal} kcal per serving (serving weight unknown, can't scale to {grams:g} g)"
        return f"{cal * grams / serving:.0f} kcal for {grams:g} g ({cal} kcal per {serving} g serving)"
    if serving:
        return f"{cal} kcal per {serving} g serving ({cal * 100 / serving:.0f} kcal/100 g)"
    return f"{cal} kcal per serving (serving weight unknown)"


@click.command()
@click.argument("image", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--model", "model_path", type=click.Path(dir_okay=False, path_type=Path),
              default=MODEL_PATH, show_default=True, help="Saved classifier from checkpoint2_baseline.py.")
@click.option("--calories", "calories_path", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=CALORIES_PATH, show_default=True, help="Calorie table JSON.")
@click.option("--top-k", type=click.IntRange(min=1), default=3, show_default=True,
              help="How many dishes to show.")
@click.option("--grams", type=click.FloatRange(min=0, min_open=True), default=None,
              help="Portion weight in grams; scales calories from the table's serving size.")
def main(image: Path, model_path: Path, calories_path: Path, top_k: int, grams: float | None) -> None:
    """Predict the dish in IMAGE and report its calories."""
    if not model_path.is_file():
        raise click.ClickException(
            f"classifier not found: {model_path}\n"
            "Run scripts/checkpoint2_baseline.py first to train and save it.")

    saved = joblib.load(model_path)
    clf, classes = saved["clf"], saved["classes"]
    table = {row["food_item"]: row
             for row in json.loads(calories_path.read_text(encoding="utf-8"))}

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    backbone, transform = load_backbone(device)

    proba = clf.predict_proba(embed(image, backbone, transform, device))[0]
    top = np.argsort(-proba)[:top_k]

    dish = classes[top[0]]
    click.secho(
        f"{dish.replace('_', ' ')}  ({proba[top[0]]:.1%} confidence)", fg="green", bold=True)
    click.echo(f"  {describe_calories(table.get(dish), grams)}")

    if len(top) > 1:
        click.echo("\nOther candidates:")
        for i in top[1:]:
            name = classes[i]
            click.echo(f"  {name.replace('_', ' '):<24} {proba[i]:6.1%}  "
                       f"{describe_calories(table.get(name), grams)}")


if __name__ == "__main__":
    main()
