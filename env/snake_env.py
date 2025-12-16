print("LOADED snake_env.py FROM:", __file__)

import random, collections
from dataclasses import dataclass
import numpy as np
from config import Config

DIRS = np.array([(0, -1), (1, 0), (0, 1), (-1, 0)], dtype=np.int32)

APPLE_NORMAL = 0
APPLE_SHRINK = 1
APPLE_GROW2  = 2

FAIL_WALL    = 0
FAIL_SELF    = 1
FAIL_TIMEOUT = 2


@dataclass
class EpisodeStats:
    ep_return: float = 0.0
    steps: int = 0


class SnakeEnv:
    def __init__(self, w: int, h: int, cfg: Config):
        self.w, self.h = w, h
        self.cfg = cfg

        self.boundary = np.zeros((h, w), dtype=np.uint8)
        self.boundary[0, :] = 1
        self.boundary[-1, :] = 1
        self.boundary[:, 0] = 1
        self.boundary[:, -1] = 1

        self.apple_normal = None
        self.apple_shrink = None
        self.apple_grow2  = None

        self.reset()

    # ------------------------
    def reset(self):
        self.done = False
        self.stats = EpisodeStats()

        cx, cy = self.w // 2, self.h // 2
        self.snake = collections.deque([
            (cx, cy),
            (cx - 1, cy),
            (cx - 2, cy),
        ])
        self.dir = 1

        self._rebuild_occupancy()
        self._compute_reachable()
        self._place_apples()

        return self.obs()

    # ------------------------
    def _rebuild_occupancy(self):
        self.occ = np.zeros((self.h, self.w), dtype=np.uint8)
        for x, y in self.snake:
            self.occ[y, x] = 1

    # ------------------------
    def _local_degree(self):
        hx, hy = self.snake[0]
        deg = 0
        for dx, dy in DIRS:
            nx, ny = hx + dx, hy + dy
            if not self.boundary[ny, nx] and not self.occ[ny, nx]:
                deg += 1
        return deg

    # ------------------------
    def _compute_reachable(self):
        mask = np.zeros((self.h, self.w), dtype=np.uint8)
        hx, hy = self.snake[0]

        body = self.occ.copy()
        tx, ty = self.snake[-1]
        body[ty, tx] = 0

        stack = [(hx, hy)]
        mask[hy, hx] = 1

        while stack:
            x, y = stack.pop()
            for dx, dy in DIRS:
                nx, ny = x + dx, y + dy
                if nx < 0 or ny < 0 or nx >= self.w or ny >= self.h:
                    continue
                if mask[ny, nx]:
                    continue
                if self.boundary[ny, nx] or body[ny, nx]:
                    continue
                mask[ny, nx] = 1
                stack.append((nx, ny))

        self._reach_mask = mask
        self._tail_reachable = bool(mask[ty, tx])

    # ------------------------
    def _place_apples(self):
        empties = np.where((self.boundary == 0) & (self.occ == 0))
        cells = list(zip(empties[1], empties[0]))
        if not cells:
            return

        self.apple_normal = random.choice(cells)
        self.apple_shrink = None
        self.apple_grow2  = None

        reach_frac = np.sum(self._reach_mask) / float(self.w * self.h)
        deg        = self._local_degree()
        tail_ok    = self._tail_reachable
        length     = len(self.snake)

        pressure = (
            length >= 12 and (
                not tail_ok or
                reach_frac < 0.35 or
                deg <= 1
            )
        )

        if pressure:
            opts = [c for c in cells if c != self.apple_normal]
            if opts:
                self.apple_shrink = random.choice(opts)
            return

        healthy = (
            tail_ok and
            reach_frac > 0.50 and
            deg >= 3 and
            length <= 14
        )

        if healthy and random.random() < 0.10:
            opts = [c for c in cells if c != self.apple_normal]
            if opts:
                self.apple_grow2 = random.choice(opts)

    # ------------------------
    def obs(self):
        self._compute_reachable()

        grid = np.zeros((10, self.h, self.w), dtype=np.float32)
        grid[0] = self.boundary

        if self.apple_normal:
            ax, ay = self.apple_normal
            grid[1, ay, ax] = 1.0

        if self.apple_shrink:
            sx, sy = self.apple_shrink
            grid[2, sy, sx] = 1.0

        if self.apple_grow2:
            gx, gy = self.apple_grow2
            grid[3, gy, gx] = 1.0

        hx, hy = self.snake[0]
        grid[4, hy, hx] = 1.0

        for x, y in list(self.snake)[1:]:
            grid[5, y, x] = 1.0

        grid[6] = self._reach_mask
        grid[7].fill(1.0 if self._tail_reachable else 0.0)
        grid[8].fill(self._local_degree()/ 4.0)

        # current local degree
        deg_now = self._local_degree()

        # projected degree if we move one step forward
        future_deg = 0
        hx, hy = self.snake[0]
        dx, dy = DIRS[self.dir]
        nx, ny = hx + dx, hy + dy

        if 0 <= nx < self.w and 0 <= ny < self.h:
            if not self.boundary[ny, nx] and not self.occ[ny, nx]:
                for dx2, dy2 in DIRS:
                    tx, ty = nx + dx2, ny + dy2
                    if 0 <= tx < self.w and 0 <= ty < self.h:
                        if not self.boundary[ty, tx] and not self.occ[ty, tx]:
                            future_deg += 1

        # encode change in degree (clamped to [-1, 1] range by /4)
        grid[9].fill((future_deg - deg_now) / 4.0)
        return grid

    # ------------------------
    def step(self, action: int):
        if self.done:
            return self.obs(), 0.0, True, {}

        if action is not None and (action + 2) % 4 != self.dir:
            self.dir = action

        dx, dy = DIRS[self.dir]
        hx, hy = self.snake[0]
        nx, ny = hx + dx, hy + dy

        reward = self.cfg.reward_step
        self.stats.steps += 1

        # ----- LOW-DEGREE PENALTY (late game only) -----
        if len(self.snake) >= 12:
            next_deg = 0
            for dx2, dy2 in DIRS:
                tx, ty = nx + dx2, ny + dy2

                if tx < 0 or ty < 0 or tx >= self.w or ty >= self.h:
                    continue

                if not self.boundary[ty, tx] and not self.occ[ty, tx]:
                    next_deg += 1

            # Penalize stepping into tight cells
            if next_deg <= 1:
                reward -= 0.01

        if self.apple_normal:
            ax, ay = self.apple_normal
            old = abs(hx - ax) + abs(hy - ay)
            new = abs(nx - ax) + abs(ny - ay)
            if self._reach_mask[ay, ax] and not self.boundary[ny, nx] and not self.occ[ny, nx]:
                if new < old:
                    reward += 0.004

        if self.boundary[ny, nx]:
            return self.obs(), reward + self.cfg.reward_die, True, {"fail_code": FAIL_WALL}
        if self.occ[ny, nx]:
            return self.obs(), reward + self.cfg.reward_die, True, {"fail_code": FAIL_SELF}

        self.snake.appendleft((nx, ny))
        self.occ[ny, nx] = 1

        if (nx, ny) == self.apple_normal:
            reward += self.cfg.reward_apple
            self._place_apples()

        elif (nx, ny) == self.apple_shrink:
            reward += self.cfg.reward_apple - 1.0
            for _ in range(2):
                if len(self.snake) > 3:
                    tx, ty = self.snake.pop()
                    self.occ[ty, tx] = 0
            self._place_apples()

        elif (nx, ny) == self.apple_grow2:
            reward += self.cfg.reward_apple
            for _ in range(2):
                self.snake.append(self.snake[-1])
                self.occ[self.snake[-1][1], self.snake[-1][0]] = 1
            self._place_apples()

        else:
            tx, ty = self.snake.pop()
            self.occ[ty, tx] = 0

        if self.stats.steps >= self.cfg.max_steps_per_episode:
            return self.obs(), reward, True, {"fail_code": FAIL_TIMEOUT}

        self.stats.ep_return += reward
        return self.obs(), reward, False, {}
