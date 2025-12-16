import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import time, random, argparse, collections
import numpy as np
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
import pygame

from env.async_arenas import AsyncArenas
from config import Config
from rl.model import MultiHeadDQN
from rl.replay import ReplayBuffer
from rl.train import train_tick, epsilon_by_step
from rl.checkpoints import load_checkpoint
from ui.input import EscDebouncer
from ui.menus import startup_menu, pause_menu, draw_text
from ui.render_run import compute_run_window_size, render_run_mode
from analysis.failures import make_run_failure_trackers, print_run_failure_summary

# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def run_tick(cfg, model, run_pack, fail_counts, fail_loss):
    # Observations: (N, 9, H, W)
    obs_np = run_pack.obs
    obs_t = torch.from_numpy(obs_np).to(cfg.device, non_blocking=True)

    # REQUIRED: arena_ids
    n = obs_t.shape[0]
    arena_ids = torch.arange(n, device=cfg.device, dtype=torch.long)

    # Forward pass (matches training contract)
    q = model(obs_t, arena_ids)
    if isinstance(q, (tuple, list)):
        q = q[0]

    actions = torch.argmax(q, dim=-1).to(torch.int64)
    actions_np = actions.cpu().numpy()

    run_pack.step_async(actions_np)
    obs2, rew, done = run_pack.step_wait()

 # ----- FAILURE TRACKING (ASYNC-SAFE) -----
    done_idx = np.nonzero(done)[0]
    for i in done_idx:
        tag = int(run_pack.fail[i])
        if tag != 255:  # 255 = sentinel = no failure
            fail_counts[tag] += 1

        # optional: estimate loss relative to expected return
        loss = cfg.run_expected_return_for_loss - float(rew[i])
        fail_loss[tag] += loss
    """
    Minimal RUN tick:
    - Reads current observations from run_pack.shared obs buffer
    - Computes greedy actions from the model
    - Steps AsyncArenas once
    - Updates failure trackers if classify_failure supports it (best-effort, no hard dependency)
    """
    # obs: (N, 9, H, W) float32 in shared memory
    obs_np = run_pack.obs  # zero-copy view
    obs_t = torch.from_numpy(obs_np).to(cfg.device, non_blocking=True)



# -----------------------------
def main():
    train_pack = None
    run_pack = None

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--grid", type=int, default=15)
    parser.add_argument("--train_arenas", type=int, default=64)
    parser.add_argument("--run_arenas", type=int, default=4)
    args = parser.parse_args()

    cfg = Config(
        grid_w=args.grid,
        grid_h=args.grid,
        train_arenas=args.train_arenas,
        run_arenas=args.run_arenas
    )

    set_seed(args.seed)

    pygame.init()
    pygame.display.set_caption("Snake RL (Forge) — Modular")
    font = pygame.font.SysFont("consolas", 18)
    clock = pygame.time.Clock()
    esc = EscDebouncer(cfg.esc_debounce_ms)

    screen = pygame.display.set_mode((640, 480))

    model = MultiHeadDQN(cfg.train_arenas, cfg.grid_w, cfg.grid_h).to(cfg.device)
    target = MultiHeadDQN(cfg.train_arenas, cfg.grid_w, cfg.grid_h).to(cfg.device)
    target.load_state_dict(model.state_dict())
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    replay = ReplayBuffer(cfg.replay_size, (10, cfg.grid_h, cfg.grid_w), cfg.device)

    rolling = collections.deque(maxlen=cfg.rolling_window)
    best_avg = -1e9

    meta = load_checkpoint(cfg, model, None, cfg.best_name)
    if meta:
        best_avg = float(meta.get("best_avgR", best_avg))
        print(f"[load] loaded checkpoint avgR={best_avg:.3f}")
    
    target.load_state_dict(model.state_dict())
    choice = startup_menu(screen, cfg)
    if choice == "QUIT":
        pygame.quit()
        return

   

    train_pack = AsyncArenas(
    w=cfg.grid_w, h=cfg.grid_h, cfg=cfg,
    total_envs=cfg.train_arenas,
    envs_per_worker=4
    )

    run_pack = AsyncArenas(
    w=cfg.grid_w, h=cfg.grid_h, cfg=cfg,
    total_envs=cfg.run_arenas,
    envs_per_worker=1
    )

    def shutdown():
        nonlocal train_pack, run_pack
        if train_pack is not None:
            try:
                train_pack.close()
            except Exception:
                pass
        if run_pack is not None:
            try:
                run_pack.close()
            except Exception:
                pass
        pygame.quit()

    mode = choice
    paused = False

    global_step = 0
    total_env_steps = 0
    run_global_steps = 0
    last_status_t = time.time()

    fail_counts, fail_loss = make_run_failure_trackers()

    if mode == "RUN":
        load_checkpoint(cfg, model, None, cfg.best_name)
        screen = pygame.display.set_mode(compute_run_window_size(cfg))
    else:
        screen = pygame.display.set_mode((520, 180))

    print(f"[mode] {mode} (device={cfg.device})")

    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                shutdown()
                return
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                if esc.pressed():
                    paused = not paused

        if paused:
            action = pause_menu(screen, cfg, mode)
            if action == "QUIT":
                shutdown()
                return
            if action == "RESUME":
                paused = False
            elif action == "SWITCH_TO_RUN":
                load_checkpoint(cfg, model, None, cfg.best_name)
                mode = "RUN"
                paused = False
                run_global_steps = 0
                fail_counts.clear()
                fail_loss.clear()
                screen = pygame.display.set_mode(compute_run_window_size(cfg))
                print("[mode] RUN")
            elif action == "SWITCH_TO_TRAIN":
                mode = "TRAIN"
                paused = False
                screen = pygame.display.set_mode((520, 180))
                print("[mode] TRAIN")
            continue

        if mode == "TRAIN":
            global_step, total_env_steps, best_avg, avgR = train_tick(
                cfg, model, target, optim, replay, train_pack,
                global_step, total_env_steps, rolling, best_avg
            )

            now = time.time()
            if now - last_status_t > 1.0:
                last_status_t = now
                epsv = epsilon_by_step(cfg, total_env_steps)
                print(
                    f"[train] env_steps={total_env_steps} "
                    f"eps={epsv:.3f} avgR({len(rolling)}ep)={avgR:.3f} "
                    f"best={best_avg:.3f} replay={replay.size}"
                )

            screen.fill((18, 18, 22))
            draw_text(screen, font, "TRAINING (ESC = pause/menu)", 20, 20)
            draw_text(screen, font, f"env_steps: {total_env_steps}", 20, 50)
            draw_text(
                screen,
                font,
                f"rolling avgR: {avgR:.3f}  best: {best_avg:.3f}",
                20,
                80
            )
            draw_text(screen, font, "Console shows detailed status.", 20, 110)
            pygame.display.flip()
            clock.tick(60)

        else:  # RUN
            run_tick(cfg, model, run_pack, fail_counts, fail_loss)
            run_global_steps += 1

            if run_global_steps % cfg.run_log_every_steps == 0:
                print_run_failure_summary(cfg, fail_counts, fail_loss)

            render_run_mode(screen, font, cfg, run_pack, model, run_global_steps)
            clock.tick(cfg.fps_run)


if __name__ == "__main__":
    main()
