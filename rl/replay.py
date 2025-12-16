import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape, device: str):
        self.capacity = capacity
        self.device = device

        self.obs = np.zeros((capacity,) + obs_shape, dtype=np.float32)
        self.next_obs = np.zeros((capacity,) + obs_shape, dtype=np.float32)
        self.act = np.zeros((capacity,), dtype=np.int64)
        self.rew = np.zeros((capacity,), dtype=np.float32)
        self.done = np.zeros((capacity,), dtype=np.float32)
        self.arena = np.zeros((capacity,), dtype=np.int64)

        self.idx = 0
        self.size = 0

    def push(self, o, a, r, no, d, arena_id: int):
        self.obs[self.idx] = o
        self.next_obs[self.idx] = no
        self.act[self.idx] = a
        self.rew[self.idx] = r
        self.done[self.idx] = float(d)
        self.arena[self.idx] = arena_id
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idxs = np.random.randint(0, self.size, size=batch_size)
        obs = torch.from_numpy(self.obs[idxs]).to(self.device)
        next_obs = torch.from_numpy(self.next_obs[idxs]).to(self.device)
        act = torch.from_numpy(self.act[idxs]).to(self.device)
        rew = torch.from_numpy(self.rew[idxs]).to(self.device)
        done = torch.from_numpy(self.done[idxs]).to(self.device)
        arena = torch.from_numpy(self.arena[idxs]).to(self.device)
        return obs, act, rew, next_obs, done, arena
