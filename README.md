# Snake RL — Reinforcement Learning Sandbox

A research-oriented Snake reinforcement-learning environment built with PyTorch.

This project implements a reinforcement-learning Snake agent designed to study learning behavior, reward shaping, and control dynamics under delayed consequences. The agent learns purely from numerical rewards and penalties, without scripted rules or hard-coded strategies.

The repository is intended as an experimental sandbox for observing emergent behavior and failure modes in reinforcement learning.

---

## What This Is

- A reinforcement-learning Snake environment
- A testbed for studying:
  - delayed reward effects
  - policy collapse
  - reward exploitation (looping, stalling)
  - survival vs. reward tradeoffs
- A compact environment where small reward changes produce large behavioral shifts

---

## What This Is Not

❌ A scripted or rule-based Snake bot  
❌ A shortest-path solver  
❌ A benchmark or “perfect” Snake agent  

The agent does not know explicit rules like “avoid walls.”  
It updates its policy solely through gradient-based learning from outcomes.

---

## Learning Loop

1. Observe environment state  
2. Predict action values  
3. Select an action (with exploration)  
4. Receive reward or penalty  
5. Compute prediction error  
6. Update network weights  

This process allows the agent to:
- learn from failure
- propagate consequences backward
- trade short-term reward for long-term survival
- exploit reward structures when misaligned

---

## Why Snake?

Snake is well-suited for reinforcement learning research because it combines:
- simple mechanics
- a large state space
- delayed consequences
- clear and observable failure modes

This makes it an effective environment for studying learning dynamics and control behavior.

---

## Key Observations

- Learning behavior emerges from reward structure
- Optimization alone does not guarantee stability
- Control mechanisms influence long-term performance
- Misaligned incentives produce predictable failure patterns

---

## Running

```bash
python main.py
