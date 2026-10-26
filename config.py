"""Typed config: config.yaml -> Config, used as cfg.train.lr (autocomplete + validation)."""

from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Section(BaseModel):
    model_config = ConfigDict(extra="forbid")  # a typo in the YAML is an error, not silently ignored


class DataConfig(Section):
    root: str
    window: int = Field(1200, gt=0)
    train_stride: int = Field(600, gt=0)
    eval_stride: int = Field(1200, gt=0)
    gap: int = Field(1200, ge=0)
    train_frac: float = Field(0.70, gt=0)
    val_frac: float = Field(0.15, gt=0)

    @model_validator(mode="after")
    def leave_room_for_test(self):
        if self.train_frac + self.val_frac >= 1:
            raise ValueError("train_frac + val_frac must be < 1 to leave a test split")
        return self


class ModelConfig(Section):
    widths: list[int] = [32, 64, 128]
    first_kernel: int = Field(64, gt=0)
    first_stride: int = Field(8, gt=0)
    dropout: float = Field(0.3, ge=0, lt=1)


class TrainConfig(Section):
    seed: int = 42
    output_dir: str = "runs/cwru-cnn1d"
    epochs: int = Field(30, gt=0)
    batch_size: int = Field(64, gt=0)
    num_workers: int = Field(0, ge=0)
    lr: float = Field(1e-3, gt=0)
    weight_decay: float = Field(0.01, ge=0)
    warmup_steps: int = Field(100, gt=0)
    label_smoothing: float = Field(0.0, ge=0, lt=1)
    grad_clip: float = Field(1.0, gt=0)
    precision: Literal["fp32", "fp16", "bf16"] = "fp32"
    grad_accum_steps: int = Field(1, gt=0)


class WandbConfig(Section):
    enabled: bool = False
    project: str = "cwru-cnn1d"
    run_name: str | None = None


class Config(Section):
    data: DataConfig
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()
    wandb: WandbConfig = WandbConfig()


def load_config(path="config.yaml") -> Config:
    with open(path, encoding="utf-8") as f:
        return Config.model_validate(yaml.safe_load(f))
