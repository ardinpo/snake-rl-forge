import torch
import pygame
import numpy as np
from config import Config
from env.snake_env import APPLE_SHRINK
from ui.menus import draw_text

RUN_HEADER_H = 48
RUN_LABEL_H = 22
RUN_MARGIN = 20

def compute_run_window_size(cfg: Config):
    scale = cfg.win_scale
    arena_w = cfg.grid_w * scale
    arena_h = cfg.grid_h * scale
    width = 2 * arena_w + 3 * RUN_MARGIN
    height = RUN_HEADER_H + 2 * (arena_h + RUN_LABEL_H) + 3 * RUN_MARGIN
    return width, height

def render_run_mode(screen, font, cfg, arena_pack, model, run_global_steps):
    screen.fill((10, 10, 14))

    obs = arena_pack.obs
    n_show = min(4, obs.shape[0])
    
    with torch.no_grad():
        o_t = torch.from_numpy(obs[:n_show]).to(cfg.device)
        arena_ids = torch.arange(n_show, device=cfg.device)
        q = model(o_t, arena_ids=arena_ids)# [N,4]
        top2 = torch.topk(q, k=2, dim=1).values
        q_margin = (top2[:, 0] - top2[:, 1]).cpu().numpy()
        # Normalize for visualization (clip extreme confidence)
    
    conf = np.clip(q_margin / 0.5, 0.0, 1.0)
    print(f"q_margin[0]={q_margin[0]:.3f}  conf[0]={conf[0]:.2f}")
    

    avail_w = screen.get_width() - 20
    cell = max(4, avail_w // (cfg.grid_w * 4))
    pad = 10

    avail_w = screen.get_width() - 20
    cell = max(4, avail_w // (cfg.grid_w * 4))
    for i in range(n_show):
        o = obs[i]

        ox = pad + i * (cfg.grid_w * cell + pad)
        oy = pad

        # channels (per your SnakeEnv contract)
        wall  = o[0]
        apple_normal = o[1]
        apple_shrink =o[2]
        apple_grow2  = o[3]
        head  = o[4]
        body  = o[5]

        for y in range(cfg.grid_h):
            for x in range(cfg.grid_w):
                px = ox + x * cell
                py = oy + y * cell

                if wall[y, x] > 0:
                    color = (80, 80, 80)
                elif body[y, x] > 0:
                    color = (0, 160, 0)
                elif head[y, x] > 0:
                    color = (0, 255, 0)
                elif apple_normal [y, x] > 0:
                    color = (220, 40, 40)
                elif apple_shrink[y, x] > 0:
                    color = (80, 200, 255)   # blue-ish
                elif apple_grow2[y, x] > 0:
                    color = (255, 180, 40)   # gold-ish 
                else:
                    continue

                pygame.draw.rect(screen, color, (px, py, cell, cell))
                pygame.draw.rect(
                    screen,
                    (255, 0, 0),
                    (px, py, cell, cell),
    1  # outline only
                )
                # ------------------------------
# Indecision overlay (red = unsure)
# ------------------------------
            alpha = int((1.0 - conf[i]) * 120) # stronger red when unsure
            if alpha > 0:
                overlay = pygame.Surface((cell, cell), pygame.SRCALPHA)
                overlay.fill((255, 0, 0, alpha))
                screen.blit(overlay, (px, py))

    draw_text(
        screen, font,
        f"RUN  steps={run_global_steps}",
        10, cfg.grid_h * cell + 2 * pad
    )

    pygame.display.flip()
