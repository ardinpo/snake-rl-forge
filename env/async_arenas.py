# env/async_arenas.py
import multiprocessing as mp
import numpy as np
from multiprocessing.shared_memory import SharedMemory

def _worker(rank, cfg_bytes, w, h, envs_per_worker,
            obs_name, rew_name, done_name, fail_name,
            act_q, ack_q):
    from config import Config
    from env.snake_env import SnakeEnv

    cfg = Config.from_bytes(cfg_bytes)
    envs = [SnakeEnv(w, h, cfg) for _ in range(envs_per_worker)]

    shm_obs  = SharedMemory(name=obs_name)
    shm_rew  = SharedMemory(name=rew_name)
    shm_done = SharedMemory(name=done_name)
    shm_fail = SharedMemory(name=fail_name)
    
    fail = np.ndarray((cfg.total_envs,), np.uint8, buffer=shm_fail.buf)
    obs  = np.ndarray((cfg.total_envs, 10, h, w), np.float32, buffer=shm_obs.buf)
    rew  = np.ndarray((cfg.total_envs,), np.float32, buffer=shm_rew.buf)
    done = np.ndarray((cfg.total_envs,), np.uint8,   buffer=shm_done.buf)

    base = rank * envs_per_worker

    while True:
        actions = act_q.get()
        if actions is None:
            break

        for i, e in enumerate(envs):
            o, r, d, info = e.step(int(actions[i]))

            if d:
                # record failure before reset
                fail[base + i] = info.get("fail_code", 255)
                o = e.reset()
            else:
                fail[base + i] = 255

            obs[base + i] = o
            rew[base + i] = r
            done[base + i] = 1 if d else 0

        ack_q.put(rank)

    shm_obs.close()
    shm_rew.close()
    shm_done.close()


class AsyncArenas:
    def __init__(self, *, w, h, cfg, total_envs, envs_per_worker):
        assert total_envs % envs_per_worker == 0

        self.total_envs = total_envs
        self.envs_per_worker = envs_per_worker
        self.n_workers = total_envs // envs_per_worker

        cfg.total_envs = total_envs
        cfg_bytes = cfg.to_bytes()

        self.obs_shape = (total_envs, 10, h, w)
        self._shm_fail = SharedMemory(create=True, size=np.zeros((total_envs,), np.uint8).nbytes)
        self.fail = np.ndarray((total_envs,), np.uint8, buffer=self._shm_fail.buf)
        self.fail.fill(255)  # sentinel: no failure
        self._shm_obs  = SharedMemory(create=True, size=np.zeros(self.obs_shape, np.float32).nbytes)
        self._shm_rew  = SharedMemory(create=True, size=np.zeros((total_envs,), np.float32).nbytes)
        self._shm_done = SharedMemory(create=True, size=np.zeros((total_envs,), np.uint8).nbytes)

        self.obs  = np.ndarray(self.obs_shape, np.float32, buffer=self._shm_obs.buf)
        self.rew  = np.ndarray((total_envs,), np.float32, buffer=self._shm_rew.buf)
        self.done = np.ndarray((total_envs,), np.uint8,   buffer=self._shm_done.buf)

        ctx = mp.get_context("spawn")
        self.act_qs = [ctx.Queue(maxsize=1) for _ in range(self.n_workers)]
        self.ack_q  = ctx.Queue()

        self.ps = []
        for rank in range(self.n_workers):
            p = ctx.Process(
                target=_worker,
                args=(rank, cfg_bytes, w, h, envs_per_worker,
                      self._shm_obs.name,
                      self._shm_rew.name,
                      self._shm_done.name,
                      self._shm_fail.name,
                      self.act_qs[rank],
                      self.ack_q),
                daemon=True
            )
            p.start()
            self.ps.append(p)

    def step_async(self, actions_np):
        for rank in range(self.n_workers):
            b = rank * self.envs_per_worker
            self.act_qs[rank].put(actions_np[b:b+self.envs_per_worker])

    def step_wait(self):
        for _ in range(self.n_workers):
            self.ack_q.get()
        return self.obs, self.rew, self.done

    def close(self):
        for q in self.act_qs:
            q.put(None)
        for p in self.ps:
            p.join(timeout=1.0)
        self._shm_obs.unlink()
        self._shm_rew.unlink()
        self._shm_done.unlink()
