# Physics-Informed Reinforcement Learning (PIRL) for Hybrid Energy Management

This repository investigates **Physics-Informed Reinforcement Learning (PIRL)** for energy
management of hybrid electric powertrains. Instead of learning a black-box policy/value function
from scratch, PIRL embeds the plant's own physics directly into the actor and critic networks:

- **Physics feature layer** — the network is handed pre-computed physical quantities (required
  power, feasible speed/power windows, battery limits, …) instead of having to rediscover them
  from raw state.
- **Physics-constraint output layer** — the actor's raw output is mapped into the physically
  feasible action set for the current state, so every action — even while exploring — respects
  actuator, battery and thermal limits.
- **Physics-informed critic** — `Q(s, a) = r_phys(s, a) + γ · V_φ(f_phys(s, a))`, where
  `r_phys`/`f_phys` are the plant's differentiable reward/transition functions and only `V_φ` is a
  learned network. Gradients from the critic flow straight through the physics to the actor.

This idea is applied to three plants of increasing complexity, each a self-contained experiment:

| Folder | Plant | Physics highlights | Baselines compared |
|---|---|---|---|
| [`PIRL/`](PIRL) | Prius-style hybrid electric vehicle | Battery R_int model, differentiable engine on/off, differentiable fuel map | DDPG, Dynamic Programming (DP) |
| [`PIRL_excavator/`](PIRL_excavator) | Parallel hybrid excavator | Boom-lowering hydraulic energy recovery, battery thermal + aging model | DP, A-ECMS, Rule-based, SAC, TD3, DDPG (+ ablations) |
| [`PIRL_excavator_ps/`](PIRL_excavator_ps) | Power-split hybrid excavator | Planetary gear (carrier = ICE, sun = pump, ring = EMG), clutch-switched lifting/lowering modes | Same as above, plus PIRL-P / PIRL-P+ (physics-critic action search) |

Each folder's own `README.md` / `BENCHMARK.md` (in Vietnamese) has the full model write-up, results
tables and figures. **See [`PIRL_README.md`](PIRL_README.md) for a step-by-step guide to running
every experiment**, including exact CLI commands, expected run times, and where pre-computed
results already live in this repo.

## Quick start

```bash
pip install torch numpy scipy matplotlib
python PIRL/pirl_hev.py
```

That trains one PIRL agent on the HEV plant and evaluates it against a rule-based baseline —
the fastest way to see the approach work end to end. `PIRL_README.md` covers the full benchmark
suites (PIRL vs. DDPG/SAC/TD3/A-ECMS/DP, ablations, plant-mismatch robustness tests) for all three
plants.

## Repository layout

```
PIRL/                            HEV plant, PIRL agent, DDPG baseline, DP benchmark
PIRL_excavator/                  Parallel hybrid excavator plant + full benchmark suite
PIRL_excavator_ps/                Power-split hybrid excavator plant + full benchmark suite
Data_Standard Driving Cycles/    Standard drive-cycle data (speed vs. time) used by PIRL/
PIRL_README.md                   Step-by-step run guide for all three experiments
```

## Origins / acknowledgment

The HEV plant, the Prius drive-cycle data, and the general "expert-knowledge-assisted DRL" idea
build on the rule-interposing DRL (RIDRL) energy-management work originally published in this
repository:

**[Lian R, Peng J, Wu Y, et al. Rule-interposing deep reinforcement learning based energy
management strategy for power-split hybrid electric vehicle. Energy, 2020:
117297.](https://www.sciencedirect.com/science/article/pii/S0360544220304047)** — original code by
[Renzong Lian](https://github.com/lryz0612) and [Yuankai Wu](https://github.com/Kaimaoge)
([original repository](https://github.com/lryz0612/Deep-reinforcement-learning-based-energy-management-strategy-for-hybrid-electric-vehicle)).

PIRL takes that idea further: instead of interposing hand-written rules or an optimal-BSFC curve
as auxiliary guidance, the plant's physics is embedded directly into the network architecture and
the critic's Bellman target, so exploration is safe by construction and gradients carry real
physical structure.
