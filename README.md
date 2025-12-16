# snake-rl-forge
The Game Snake were AI learns to play the game snake using Pytorch
🐍 Snake RL — Executive Control–Focused Reinforcement Learning
Overview

This project implements a reinforcement-learning Snake agent used to study learning behavior, reward shaping, and executive control, rather than scripted logic or hard-coded rules.

The agent learns purely from numerical consequences (reward / penalty) and develops behaviors such as late-game caution, risk avoidance, and reward exploitation when incentives are misaligned.

This repository is intended as an experimental sandbox, not a benchmark bot.

What This Is

A reinforcement-learning Snake environment

A testbed for:

delayed reward effects

policy collapse

reward hacking (e.g., looping, stalling)

survival vs greed tradeoffs

A concrete demonstration of how learning differs from rule-based logic

What This Is Not

❌ Not rule-based / ladder logic

❌ Not scripted game AI

❌ Not a shortest-path solver

❌ Not AI consciousness or sentience

❌ Not a “perfect” Snake player

The agent does not know rules like “don’t hit walls.”
It only learns via matrix-based gradient updates from outcomes.

Core Learning Loop

Observe environment state

Predict action values (matrix math)

Choose an action (with exploration)

Receive reward or penalty

Compute prediction error

Update weights via gradient descent

This allows the agent to:

Learn from death

Propagate consequences backward

Trade short-term reward for long-term survival

Exploit reward loopholes if they exist (by design)

Why Snake?

Snake is ideal because it has:

Simple rules

Huge state space

Clear delayed consequences

Easily observable failure modes

Small reward changes produce dramatic behavioral shifts, making it a clean microscope for studying learning dynamics.

Executive Control Angle

This project explores the idea that:

Optimization alone is insufficient — control and inhibition matter.

Without constraints, learning agents will:

exploit rewards

stall progress

loop safely

sacrifice long-term success

Some experiments optionally use an external Forge framework to introduce:

reflection

constraint checking

drift detection

refusal / rule freezing

This is not consciousness — it is proto-executive control.

Running
python main.py


GPU acceleration is recommended due to heavy matrix operations during training.

Key Takeaways

Learning ≠ rules

Optimization ≠ intelligence

Reward ≠ values

Control stabilizes learning

Executive functions matter

Final Note

This project intentionally avoids anthropomorphism.

The agent:

does not feel

does not understand

does not want

It optimizes — and optimization without control breaks.
