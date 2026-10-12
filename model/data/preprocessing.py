import json
import random
import shutil
from pathlib import Path

import click

"""

80/20 split

"""


DATA_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGES = DATA_DIR / "food-101" / "images"
DEFAULT_OUT = DATA_DIR / "splits"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def list_classes(images_dir: Path):
    return sorted(p for p in images_dir.iterdir() if p.is_dir())


def split_class(files, train_frac, rng):
    files = sorted(files)
    rng.shuffle(files)
    n_train = round(len(files) * train_frac)
    n_train = min(max(n_train, 1), len(files) -
                  1) if len(files) > 1 else len(files)
    return files[:n_train], files[n_train:]


@click.command()
@click.option("--images-dir", type=click.Path(file_okay=False, path_type=Path),
              default=DEFAULT_IMAGES, show_default=True, help="Directory of class subfolders.")
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path),
              default=DEFAULT_OUT, show_default=True, help="Where to write the split output.")
@click.option("--train-frac", type=click.FloatRange(0.0, 1.0, min_open=True, max_open=True),
              default=0.8, show_default=True, help="Fraction of each class used for training.")
@click.option("--copy", "do_copy", is_flag=True,
              help="Also copy images into out-dir/train/<class>/ and out-dir/test/<class>/.")
def main(images_dir: Path, out_dir: Path, train_frac: float, do_copy: bool):
    if not images_dir.is_dir():
        raise click.ClickException(f"Images directory not found: {images_dir}")

    class_dirs = list_classes(images_dir)
    if not class_dirs:
        raise click.ClickException(
            f"No class directories found in {images_dir}")

    classes = [p.name for p in class_dirs]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    rng = random.Random(100)

    train, test = [], []
    for cdir in class_dirs:
        files = [f for f in cdir.iterdir() if f.suffix.lower() in IMAGE_EXTS]
        if not files:
            click.secho(
                f"Warning: no images in {cdir.name}, skipping", fg="yellow")
            continue
        tr, te = split_class(files, train_frac, rng)
        label = class_to_idx[cdir.name]
        # store paths relative to DATA_DIR, posix-style, so the manifest is portable
        train += [{"path": f.relative_to(DATA_DIR).as_posix(),
                   "label": label} for f in tr]
        test += [{"path": f.relative_to(DATA_DIR).as_posix(),
                  "label": label} for f in te]

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "train_frac": train_frac,
        "classes": classes,
        "class_to_idx": class_to_idx,
        "train": train,
        "test": test,
    }
    manifest_path = out_dir / "split.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    total = len(train) + len(test)
    click.echo(f"{len(classes)} classes, {total} images")
    click.echo(
        f"train: {len(train)} ({len(train) / total:.1%})   test: {len(test)} ({len(test) / total:.1%})")
    click.secho(f"Manifest written to {manifest_path}", fg="green")

    if do_copy:
        for split_name, items in (("train", train), ("test", test)):
            with click.progressbar(items, label=f"Copying {split_name}") as bar:
                for item in bar:
                    src = DATA_DIR / item["path"]
                    dst = out_dir / split_name / src.parent.name / src.name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
        click.secho(
            f"Copied images into {out_dir / 'train'} and {out_dir / 'test'}", fg="green")


if __name__ == "__main__":
    main()
