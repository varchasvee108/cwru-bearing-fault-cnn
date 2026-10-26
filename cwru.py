"""CWRU bearing data: load .mat files, split in time, window, normalize, make DataLoaders."""

from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from scipy.signal import decimate
from torch.utils.data import DataLoader, TensorDataset

CWRU_FILES = {
    "normal": [97, 98, 99, 100],
    "inner_007": [105, 106, 107, 108],
    "ball_007": [118, 119, 120, 121],
    "outer_007": [130, 131, 132, 133],
    "inner_014": [169, 170, 171, 172],
    "ball_014": [185, 186, 187, 188],
    "outer_014": [197, 198, 199, 200],
    "inner_021": [209, 210, 211, 212],
    "ball_021": [222, 223, 224, 225],
    "outer_021": [234, 235, 236, 237],
}


def load_signal(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing CWRU file: {path}. Put the numbered .mat files in data.root.")
    mat = loadmat(path)
    key = next((k for k in mat if k.endswith("_DE_time")), None)
    if key is None:
        raise ValueError(f"{path} has no MATLAB variable ending in '_DE_time'")
    signal = np.asarray(mat[key]).ravel().astype(np.float32)
    if signal.size == 0 or not np.isfinite(signal).all():
        raise ValueError(f"{path}: empty signal or NaN/inf values")
    return signal


def make_windows(signal, size, stride):
    if len(signal) < size:
        return np.empty((0, size), dtype=np.float32)
    n = (len(signal) - size) // stride + 1
    indices = np.arange(size)[None, :] + stride * np.arange(n)[:, None]
    return signal[indices].astype(np.float32)


def prepare_data(cfg):
    """Split each recording in time BEFORE windowing so windows never overlap across splits."""
    window, gap = cfg.window, cfg.gap
    classes = list(CWRU_FILES)
    parts = {s: ([], []) for s in ("train", "val", "test")}
    for label, name in enumerate(classes):
        for number in CWRU_FILES[name]:
            signal = load_signal(Path(cfg.root) / f"{number}.mat")
            if name == "normal":  # recorded at 48 kHz, the faults at 12 kHz
                signal = decimate(signal, 4, zero_phase=True).astype(np.float32)
            n = len(signal)
            a, b = int(n * cfg.train_frac), int(n * (cfg.train_frac + cfg.val_frac))
            pieces = {
                "train": (signal[:a], cfg.train_stride),
                "val": (signal[a + gap:b], cfg.eval_stride),
                "test": (signal[b + gap:], cfg.eval_stride),
            }
            for split, (piece, stride) in pieces.items():
                windows = make_windows(piece, window, stride)
                parts[split][0].append(windows)
                parts[split][1].append(np.full(len(windows), label, dtype=np.int64))
    arrays = {s: (np.concatenate(xs), np.concatenate(ys)) for s, (xs, ys) in parts.items()}
    if any(len(y) == 0 for _, y in arrays.values()):
        raise ValueError("An empty split: check recording lengths, window, gap and fractions")
    # Normalization statistics come from training windows only.
    mean, std = float(arrays["train"][0].mean()), float(arrays["train"][0].std())
    arrays = {s: ((x - mean) / std, y) for s, (x, y) in arrays.items()}
    return arrays, {"classes": classes, "mean": mean, "std": std}


def build_loaders(cfg, batch_size, num_workers=0):
    arrays, meta = prepare_data(cfg)
    loaders = {}
    for split, (x, y) in arrays.items():
        dataset = TensorDataset(torch.from_numpy(x).unsqueeze(1), torch.from_numpy(y))  # (N, 1, window)
        loaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=split == "train",
                                    num_workers=num_workers)
    return loaders, meta
