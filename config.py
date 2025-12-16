from dataclasses import dataclass
import torch
import pickle
@dataclass
class Config:
    # Env
    grid_w: int = 15
    grid_h: int = 15
    max_steps_per_episode: int = 4000

    # DQN
    gamma: float = 0.99
    lr: float = 2e-4
    batch_size: int = 256
    replay_size: int = 200_000
    warmup_steps: int = 20_000
    update_every: int = 4
    target_sync_every: int = 2000

    # Exploration (train only)
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 300_000

    # Arenas
    train_arenas: int = 64
    run_arenas: int = 4

    # Rolling avgR + autosave
    rolling_window: int = 200
    checkpoint_dir: str = "checkpoints"
    best_name: str = "best_modular_v004.pt"# bump when obs changes

    # Curriculum thresholds (computed from free cells)
    t1_frac: float = 0.25
    t2_frac: float = 0.50
    persist_n: int = 500
    persist_m: int = 1000

    # Moving apple scaffold (stage1 full, stage3 zero)
    apple_move_prob_base: float = 0.25
    moving_apple_bonus_base: float = 0.25

    # Random walls stress (peaks mid-stage2)
    max_wall_frac: float = 0.08
    wall_refresh_interval: int = 25

    # Rewards (base game)
    reward_apple: float = 5.0
    reward_step: float = -0.005
    reward_die: float = -1.0

    # Pygame / UI
    win_scale: int = 20
    fps_run: int = 30
    esc_debounce_ms: int = 180

    # Training speed
    train_steps_per_tick: int = 64

    # RUN-only failure logging
    run_log_every_steps: int = 1000
    run_expected_return_for_loss: float = 20.0

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    def to_bytes(self):
        return pickle.dumps(self)

    @staticmethod
    def from_bytes(b):
        return pickle.loads(b)