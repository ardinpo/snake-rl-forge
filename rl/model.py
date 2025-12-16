import torch
import torch.nn as nn

class SharedBase(nn.Module):
    def __init__(self, in_ch=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        )
        self.fc = nn.Sequential(nn.Linear(64 * 15 * 15, 512), nn.ReLU())

    def forward(self, x):
        z = self.conv(x).flatten(1)
        return self.fc(z)

class MultiHeadDQN(nn.Module):
    def __init__(self, arenas: int, grid_w: int, grid_h: int, actions: int = 4):
        super().__init__()
        self.base = SharedBase(in_ch=10)
        with torch.no_grad():
            dummy = torch.zeros(1, 10, grid_h, grid_w)
            z = self.base.conv(dummy).flatten(1)
            in_dim = z.shape[1]
        self.base.fc = nn.Sequential(nn.Linear(in_dim, 512), nn.ReLU())
        self.heads = nn.ModuleList([nn.Linear(512, actions) for _ in range(arenas)])

    def forward(self, x, arena_ids: torch.Tensor):
        h = self.base(x)
        all_q = torch.stack([head(h) for head in self.heads], dim=1)  # [B,A,4]
        idx = arena_ids.view(-1, 1, 1).expand(-1, 1, all_q.size(-1))
        return all_q.gather(1, idx).squeeze(1)
