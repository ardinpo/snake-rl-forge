import os
import torch
from config import Config

def ensure_ckpt_dir(cfg: Config):
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

def save_checkpoint(cfg: Config, model, optim, meta: dict, name: str):
    ensure_ckpt_dir(cfg)
    path = os.path.join(cfg.checkpoint_dir, name)
    torch.save({"model": model.state_dict(),
                "optim": optim.state_dict() if optim is not None else None,
                "meta": meta}, path)

def load_checkpoint(cfg: Config, model, optim, name: str):
    path = os.path.join(cfg.checkpoint_dir, name)
    if not os.path.isfile(path):
        return None

    blob = torch.load(path, map_location=cfg.device)
    model_state = blob["model"]

    # =========================================================
    # PATCH: expand conv1 input channels (9 → 10) for lookahead
    # =========================================================
    key = "base.conv.0.weight"   # first Conv2d in SharedBase
    if key in model_state:
        w = model_state[key]     # shape [32, 9, 3, 3] (old)
        if w.shape[1] == 9:
            out_c, _, k1, k2 = w.shape
            new_w = torch.zeros(out_c, 10, k1, k2, device=w.device)
            new_w[:, :] = w      # preserve learned filters
            new_w[:, ] = 0.0     # neutral init for channel 9
            model_state[key] = new_w
            print("[ckpt] expanded conv1 channels 9 → 10")
    # =========================================================

    model.load_state_dict(model_state)

    if optim is not None and blob.get("optim") is not None:
        optim.load_state_dict(blob["optim"])

    return blob.get("meta", {})
