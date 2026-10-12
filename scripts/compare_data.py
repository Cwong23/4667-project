import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import click

SCRIPTS_DIR = Path(__file__).resolve().parent
DEFAULT_JSON = SCRIPTS_DIR / "food_calories.json"
DEFAULT_IMAGES = (SCRIPTS_DIR / ".." / "model" / "data" /
                  "food-101" / "images").resolve()


def normalize(name: str) -> str:
    return re.sub(r"\s+", "_", name.strip().lower())


def resolve_duplicates(foods, strategy):
    groups = defaultdict(list)
    for item in foods:
        groups[item["food_item"]].append(item)
    dupes = {n: g for n, g in groups.items() if len(g) > 1}

    result = []
    for name, group in groups.items():
        if len(group) == 1 or strategy == "first":
            result.append(group[0])
        elif strategy == "last":
            result.append(group[-1])
        elif strategy == "average":
            result.append({
                **group[0],
                "grams": round(sum(g["grams"] for g in group) / len(group)),
                "calories": round(sum(g["calories"] for g in group) / len(group)),
            })
    return result, dupes


@click.command()
@click.option("--json-path", type=click.Path(dir_okay=False, path_type=Path),
              default=DEFAULT_JSON, show_default=True, help="Path to food_calories.json.")
@click.option("--images-dir", type=click.Path(file_okay=False, path_type=Path),
              default=DEFAULT_IMAGES, show_default=True, help="Food-101 images directory.")
@click.option("--on-duplicate", type=click.Choice(["first", "last", "average", "error"]),
              default="error", show_default=True,
              help="How to handle multiple entries with the same normalized name.")
@click.option("--dry-run", is_flag=True, help="Show what would change without modifying anything.")
@click.option("--yes", "-y", is_flag=True, help="Apply changes without asking for confirmation.")
@click.option("--no-backup", is_flag=True, help="Don't write a .bak copy before overwriting the JSON.")
def main(json_path, images_dir, on_duplicate, dry_run, yes, no_backup):
    if not images_dir.is_dir():
        raise click.ClickException(f"Images directory not found: {images_dir}")

    with open(json_path, "r", encoding="utf-8") as f:
        foods = json.load(f)
    for item in foods:
        item["food_item"] = normalize(item["food_item"])

    dirs = sorted(p for p in images_dir.iterdir() if p.is_dir())
    dir_names = {p.name for p in dirs}
    matched = [i for i in foods if i["food_item"] in dir_names]
    removed_foods = [i for i in foods if i["food_item"] not in dir_names]

    deduped, dupes = resolve_duplicates(matched, on_duplicate)
    if dupes:
        click.secho(
            f"\n{len(dupes)} food_items appear more than once:", fg="yellow")
        for name, group in dupes.items():
            click.echo(f"  {name}:")
            for g in group:
                click.echo(f"    {g['grams']} g, {g['calories']} cal")
        if on_duplicate == "error":
            raise click.ClickException(
                "Duplicates found. Re-run with --on-duplicate first|last|average, "
                "or fix them in the JSON by hand."
            )
        click.echo(f"Resolving with strategy: {on_duplicate}")

    keep_names = {i["food_item"] for i in deduped}
    dirs_to_remove = [p for p in dirs if p.name not in keep_names]

    click.echo(f"\nJSON:   {len(foods)} entries -> keeping {len(deduped)}, "
               f"dropping {len(removed_foods)} unmatched + {len(matched) - len(deduped)} duplicates")
    click.echo(f"Images: {len(dirs)} directories -> keeping {len(dirs) - len(dirs_to_remove)}, "
               f"removing {len(dirs_to_remove)}")

    if not deduped:
        raise click.ClickException(
            "No food_items match any image directory; refusing to wipe everything.")

    if dry_run:
        if dirs_to_remove:
            click.echo("Would remove directories: " +
                       ", ".join(p.name for p in dirs_to_remove))
        click.echo("\nDry run, nothing was modified.")
        return

    if not yes:
        click.confirm(
            f"\nRewrite {json_path.name} and delete {len(dirs_to_remove)} directories?", abort=True
        )

    if not no_backup:
        backup = json_path.with_suffix(json_path.suffix + ".bak")
        if backup.exists():
            click.echo(
                f"Backup {backup.name} already exists, leaving it untouched.")
        else:
            shutil.copy2(json_path, backup)
            click.echo(f"Backup written to {backup.name}")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(deduped, f, indent=4)
        f.write("\n")
    click.secho(
        f"Wrote {len(deduped)} entries to {json_path.name}", fg="green")

    for p in dirs_to_remove:
        shutil.rmtree(p)
    if dirs_to_remove:
        click.secho(f"Removed {len(dirs_to_remove)} directories.", fg="green")


if __name__ == "__main__":
    main()
