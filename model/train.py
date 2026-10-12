from pathlib import Path

import click
import torch
import torch.nn as nn

from dataset import make_dataloaders
from model import build_model, load_checkpoint, set_backbone_frozen

CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        loss_sum += criterion(out, y).item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        n += y.size(0)
    return loss_sum / n, correct / n


@click.command()
@click.option("--arch", default="resnet50", show_default=True,
              type=click.Choice(["resnet18", "resnet34", "resnet50"]))
@click.option("--epochs", default=15, show_default=True)
@click.option("--freeze-epochs", default=2, show_default=True,
              help="Train only the new classifier head for this many epochs first.")
@click.option("--batch-size", default=32, show_default=True)
@click.option("--lr", default=1e-3, show_default=True, help="Peak LR for the head; backbone uses lr/10.")
@click.option("--val-frac", default=0.1, show_default=True, help="Fraction of train held out for validation.")
@click.option("--workers", default=4, show_default=True)
@click.option("--seed", default=42, show_default=True)
def main(arch, epochs, freeze_epochs, batch_size, lr, val_frac, workers, seed):
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dl, val_dl, test_dl, classes = make_dataloaders(
        batch_size, val_frac, workers, seed)
    click.echo(f"{len(classes)} classes | train {len(train_dl.dataset)} | val {len(val_dl.dataset)} | "
               f"test {len(test_dl.dataset)} | device {device}")

    model = build_model(arch, len(classes)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

    head = list(model.fc.parameters())
    backbone = [p for n, p in model.named_parameters()
                if not n.startswith("fc.")]
    optimizer = torch.optim.AdamW(
        [{"params": backbone, "lr": lr / 10}, {"params": head, "lr": lr}], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[lr / 10, lr], total_steps=epochs * len(train_dl))

    CKPT_DIR.mkdir(exist_ok=True)
    best_val = 0.0
    for epoch in range(1, epochs + 1):
        set_backbone_frozen(model, epoch <= freeze_epochs)
        model.train()
        loss_sum, correct, n = 0.0, 0, 0
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(
                device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, enabled=(device == "cuda")):
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_sum += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += y.size(0)

        val_loss, val_acc = evaluate(model, val_dl, device, criterion)
        click.echo(f"epoch {epoch:2d}/{epochs} | train loss {loss_sum / n:.3f} acc {correct / n:.3f} | "
                   f"val loss {val_loss:.3f} acc {val_acc:.3f}")
        if val_acc > best_val:
            best_val = val_acc
            torch.save({"model": model.state_dict(), "arch": arch, "classes": classes},
                       CKPT_DIR / "best.pt")

    model, _ = load_checkpoint(CKPT_DIR / "best.pt", device)
    test_loss, test_acc = evaluate(model, test_dl, device, criterion)
    click.secho(
        f"best val acc {best_val:.3f} | TEST loss {test_loss:.3f} acc {test_acc:.3f}", fg="green")


if __name__ == "__main__":
    main()
