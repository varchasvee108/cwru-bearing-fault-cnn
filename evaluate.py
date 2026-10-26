"""Score a saved checkpoint on the test split.

    python evaluate.py --checkpoint runs/cwru-cnn1d/best.pt [--split val]
"""

import argparse
from pathlib import Path

import torch

from cnn import CNN1D
from cwru import build_loaders
from config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=Path(__file__).parent / "config.yaml")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.checkpoint, map_location=device)
    loaders, meta = build_loaders(cfg.data, cfg.train.batch_size)
    model = CNN1D(num_classes=len(meta["classes"]), **state["model_cfg"]).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    classes = len(meta["classes"])
    confusion = torch.zeros(classes, classes, dtype=torch.int64)
    with torch.no_grad():
        for x, y in loaders[args.split]:
            pred = model(x.to(device)).argmax(-1).cpu()
            confusion += torch.bincount(y * classes + pred, minlength=classes**2).reshape(classes, classes)

    tp = confusion.diag().float()
    precision = tp / confusion.sum(0).clamp_min(1)
    recall = tp / confusion.sum(1).clamp_min(1)
    print(f"{args.split} accuracy {tp.sum() / confusion.sum():.4f}")
    for i, name in enumerate(meta["classes"]):
        print(f"{name:>10}  precision {precision[i]:.3f}  recall {recall[i]:.3f}  n {int(confusion[i].sum())}")
    print("confusion matrix (rows = true, cols = predicted):")
    print(confusion)


if __name__ == "__main__":
    main()
