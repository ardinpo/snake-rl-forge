import random
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
from config import Config
from rl.checkpoints import save_checkpoint


def epsilon_by_step(cfg: Config, total_env_steps: int) -> float:
    if total_env_steps >= cfg.eps_decay_steps:
        return cfg.eps_end
    t = total_env_steps / float(cfg.eps_decay_steps)
    return cfg.eps_start + t * (cfg.eps_end - cfg.eps_start)


def train_tick(cfg: Config, model, target, optim, replay, pack,
               global_step: int, total_env_steps: int,
               rolling: deque, best_avg: float):
    device = cfg.device
    A = pack.total_envs
    model.train()

    # persistent per-env episode return tracker (attach once)
    if not hasattr(pack, "_ep_ret"):
        pack._ep_ret = np.zeros((A,), dtype=np.float32)

    for _ in range(cfg.train_steps_per_tick):
        eps = epsilon_by_step(cfg, total_env_steps)

        # ---- SNAPSHOT PRE-STEP OBS (critical) ----
        obs_prev = np.empty_like(pack.obs)
        np.copyto(obs_prev, pack.obs)  # obs_prev = s_t

        obs_t = torch.from_numpy(obs_prev).to(device, non_blocking=True)  # [A,9,H,W]

        with torch.no_grad():
            arena_ids = torch.arange(A, device=device, dtype=torch.long)
            q = model(obs_t, arena_ids=arena_ids)
            greedy = torch.argmax(q, dim=1).cpu().numpy()

        eps = 0.01  # WATCH MODE epsilon (one-line fix)

        actions = np.empty((A,), dtype=np.int64)
        for i in range(A):
            if random.random() < eps:
                actions[i] = random.randrange(4)
            else:
                actions[i] = int(greedy[i])

        # ---------------- ASYNC STEP ----------------
        pack.step_async(actions)
        next_obs, rewards, dones = pack.step_wait()  # next_obs is s_{t+1} in shared memory
        # --------------------------------------------

        # episode return accumulation from step rewards
        pack._ep_ret += rewards.astype(np.float32, copy=False)

        for i in range(A):
            replay.push(
                obs_prev[i],          # s_t  (snapshot)
                actions[i],
                rewards[i],
                next_obs[i],          # s_{t+1}
                dones[i],
                arena_id=i
            )

            if dones[i]:
                rolling.append(float(pack._ep_ret[i]))  # episode return, not terminal reward
                pack._ep_ret[i] = 0.0                   # reset tracker

        total_env_steps += A
        global_step += 1

        # ---------------- TRAIN UPDATE ----------------
        if replay.size >= cfg.warmup_steps and (global_step % cfg.update_every == 0):
            obs_b, act_b, rew_b, next_obs_b, done_b, arena_b = replay.sample(cfg.batch_size)

            q = model(obs_b, arena_ids=arena_b)
            q_a = q.gather(1, act_b.view(-1, 1)).squeeze(1)

            with torch.no_grad():
                next_q_online = model(next_obs_b, arena_ids=arena_b)
                next_a = torch.argmax(next_q_online, dim=1)
                next_q_target = target(next_obs_b, arena_ids=arena_b)
                next_q = next_q_target.gather(1, next_a.view(-1, 1)).squeeze(1)
                y = rew_b + cfg.gamma * (1.0 - done_b) * next_q

            loss = F.smooth_l1_loss(q_a, y)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optim.step()

        if global_step % cfg.target_sync_every == 0:
            target.load_state_dict(model.state_dict())

    avgR = float(np.mean(rolling)) if len(rolling) > 0 else 0.0
    if len(rolling) >= max(20, cfg.rolling_window // 4) and avgR > best_avg + 1e-6:
        best_avg = avgR
        meta = {
            "best_avgR": best_avg,
            "global_step": global_step,
            "total_env_steps": total_env_steps
        }
        save_checkpoint(cfg, model, optim, meta, cfg.best_name)
        print(f"[autosave] new best avgR={best_avg:.3f} at env_steps={total_env_steps}")

    return global_step, total_env_steps, best_avg, avgR
