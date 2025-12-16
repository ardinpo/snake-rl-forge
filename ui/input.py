import pygame

class EscDebouncer:
    def __init__(self, debounce_ms: int):
        self.debounce_ms = debounce_ms
        self.last_ms = -10_000

    def pressed(self) -> bool:
        now = pygame.time.get_ticks()
        if now - self.last_ms < self.debounce_ms:
            return False
        self.last_ms = now
        return True
