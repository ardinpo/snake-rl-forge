import pygame
from config import Config

def draw_text(screen, font, txt, x, y):
    surf = font.render(txt, True, (240, 240, 240))
    screen.blit(surf, (x, y))

def startup_menu(screen, cfg: Config) -> str:
    font = pygame.font.SysFont("consolas", 20)
    clock = pygame.time.Clock()
    sel = 0
    items = ["TRAIN", "RUN", "QUIT"]

    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return "QUIT"
            if ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_UP, pygame.K_w):
                    sel = (sel - 1) % len(items)
                elif ev.key in (pygame.K_DOWN, pygame.K_s):
                    sel = (sel + 1) % len(items)
                elif ev.key in (pygame.K_RETURN, pygame.K_SPACE):
                    return items[sel]

        screen.fill((20, 20, 24))
        draw_text(screen, font, "Snake RL (Forge) — Startup", 20, 20)
        draw_text(screen, font, "Use ↑/↓ + Enter", 20, 50)
        y = 100
        for i, it in enumerate(items):
            prefix = "▶ " if i == sel else "  "
            draw_text(screen, font, f"{prefix}{it}", 20, y)
            y += 30

        pygame.display.flip()
        clock.tick(30)

def pause_menu(screen, cfg: Config, current_mode: str) -> str:
    font = pygame.font.SysFont("consolas", 20)
    clock = pygame.time.Clock()
    items = ["RESUME", "SWITCH_TO_RUN" if current_mode == "TRAIN" else "SWITCH_TO_TRAIN", "QUIT"]
    sel = 0
    active_arenas = cfg.train_arenas if current_mode == "TRAIN" else cfg.run_arenas

    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return "QUIT"
            if ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_UP, pygame.K_w):
                    sel = (sel - 1) % len(items)
                elif ev.key in (pygame.K_DOWN, pygame.K_s):
                    sel = (sel + 1) % len(items)
                elif ev.key == pygame.K_ESCAPE:
                    return "RESUME"
                elif ev.key in (pygame.K_RETURN, pygame.K_SPACE):
                    return items[sel]

        overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        screen.blit(overlay, (0, 0))

        draw_text(screen, font, f"Paused ({current_mode})", 20, 20)
        draw_text(screen, font, "Use ↑/↓ + Enter (ESC resumes)", 20, 50)
        draw_text(screen, font, f"Active arenas: {active_arenas}", 20, 80)
        y = 100
        for i, it in enumerate(items):
            prefix = "▶ " if i == sel else "  "
            draw_text(screen, font, f"{prefix}{it}", 20, y)
            y += 30

        pygame.display.flip()
        clock.tick(30)
