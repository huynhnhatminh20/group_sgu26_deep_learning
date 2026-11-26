"""Atomic epoch-boundary checkpoints for future GENIA runs."""

import os
import json
from pathlib import Path
import random

import numpy as np
import torch


def write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    try:
        with temporary.open('w', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2)
            stream.write('\n')
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_last(path, model, optimizer, epoch, best_f1, best_epoch, stale_epochs,
              config, hashes, scheduler=None, step=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch, "step": step, "best_f1": best_f1, "best_epoch": best_epoch,
        "stale_epochs": stale_epochs, "config": config, "hashes": hashes,
        "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_last(path, model, optimizer, expected_config, expected_hashes, scheduler=None):
    # Only load a checkpoint created by this project from a trusted path.
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["config"] != expected_config or state["hashes"] != expected_hashes:
        raise ValueError("Resume config or data/embedding hashes differ")
    if (state["scheduler"] is None) != (scheduler is None):
        raise ValueError("Resume scheduler configuration differs")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(state["scheduler"])
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"] is not None:
        if not torch.cuda.is_available():
            raise ValueError("CUDA RNG state cannot be restored on CPU")
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return state
