"""Train the CNN on CWRU. Everything is configured in config.yaml.

    python train.py [--config config.yaml]
"""

import argparse
import math
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from cwru import build_loaders
from cnn import CNN1D
from config import load_config


@torch.no_grad()
def evaluate(model, loader, device, autocast):
    model.eval()
    loss = correct = n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with autocast():
            logits = model(x)
        loss += F.cross_entropy(logits.float(), y, reduction="sum").item()
        correct += (logits.argmax(-1) == y).sum().item()
        n += len(y)
    return loss / n, correct / n


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=Path(__file__).parent / "config.yaml")
    config_path = parser.parse_args().config
    cfg = load_config(config_path)
    t = cfg.train

    random.seed(t.seed)
    np.random.seed(t.seed)
    torch.manual_seed(t.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Mixed precision: fp32 = off, fp16 (CUDA only, uses a GradScaler) or bf16.
    dtype = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}[t.precision]
    if t.precision == "fp16" and device.type != "cuda":
        raise ValueError("fp16 requires CUDA; use fp32 or bf16 on CPU")
    autocast = (lambda: torch.autocast(device.type, dtype=dtype)) if dtype else nullcontext
    scaler = torch.amp.GradScaler(device.type, enabled=t.precision == "fp16")

    loaders, meta = build_loaders(cfg.data, t.batch_size, t.num_workers)
    print("windows: " + " / ".join(f"{len(l.dataset)} {s}" for s, l in loaders.items()))
    model = CNN1D(num_classes=len(meta["classes"]), **cfg.model.model_dump()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=t.lr, weight_decay=t.weight_decay)

    # LR schedule is stepped once per optimizer update: linear warmup, then cosine decay.
    accum = t.grad_accum_steps
    total_steps = t.epochs * math.ceil(len(loaders["train"]) / accum)
    def lr_scale(step):
        if step < t.warmup_steps:
            return (step + 1) / t.warmup_steps
        progress = (step - t.warmup_steps) / max(1, total_steps - t.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)

    run = None
    if cfg.wandb.enabled:
        import wandb
        run = wandb.init(project=cfg.wandb.project, name=cfg.wandb.run_name,
                         config=cfg.model_dump())

    out_dir = Path(t.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device {device} | precision {t.precision} | grad_accum {accum} | {total_steps} optimizer steps")

    best_acc, step = 0.0, 0
    for epoch in range(1, t.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loader = loaders["train"]
        run_loss = run_correct = run_n = 0
        for i, (x, y) in enumerate(loader, 1):
            x, y = x.to(device), y.to(device)
            with autocast():
                logits = model(x)
            loss = F.cross_entropy(logits.float(), y, label_smoothing=t.label_smoothing)
            scaler.scale(loss / accum).backward()
            run_loss += loss.item() * len(y)
            run_correct += (logits.argmax(-1) == y).sum().item()
            run_n += len(y)

            if i % accum == 0 or i == len(loader):  # also flush a partial last group
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                step += 1

        train_loss, train_acc = run_loss / run_n, run_correct / run_n
        val_loss, val_acc = evaluate(model, loaders["val"], device, autocast)
        lr = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch}/{t.epochs} | train loss {train_loss:.4f} acc {train_acc:.4f} "
              f"| val loss {val_loss:.4f} acc {val_acc:.4f} | lr {lr:.2e}")
        if run:
            run.log({"epoch": epoch, "train/loss": train_loss, "train/acc": train_acc,
                     "val/loss": val_loss, "val/acc": val_acc, "lr": lr}, step=step)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(), "meta": meta, "model_cfg": cfg.model.model_dump()},
                       out_dir / "best.pt")

    # Final test score from the best-validation checkpoint.
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device)["model"])
    test_loss, test_acc = evaluate(model, loaders["test"], device, autocast)
    print(f"test loss {test_loss:.4f} acc {test_acc:.4f} (best val acc {best_acc:.4f})")
    if run:
        run.summary.update({"test/loss": test_loss, "test/acc": test_acc, "best_val_acc": best_acc})
        run.finish()


if __name__ == "__main__":
    main()
