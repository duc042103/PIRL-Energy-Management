# PIRL Experiments — Reproduction Guide

This file documents three Physics-Informed Reinforcement Learning (PIRL) experiments added on top of the
original Prius DDPG repository. Each embeds the plant's physics directly into the actor and critic
networks (a physics feature layer, a physics-constraint output layer, and a critic of the form
`Q = r_phys(s,a) + γ·V_φ(f_phys(s,a))`), instead of learning a black-box `Q(s,a)`.

| Folder | Subject | Physics highlights |
|---|---|---|
| [`PIRL/`](PIRL) | Prius-style HEV | battery R_int model, engine on/off, differentiable fuel map |
| [`PIRL_excavator/`](PIRL_excavator) | Parallel hybrid excavator | boom-lowering energy recovery, battery thermal + aging model |
| [`PIRL_excavator_ps/`](PIRL_excavator_ps) | Power-split hybrid excavator | planetary gear (carrier=ICE, sun=pump, ring=EMG), clutch-switched recovery mode |

Each experiment's own `README.md`/`BENCHMARK.md` (in Vietnamese) has the full write-up, figures and
analysis. This file only covers **how to run the code**.

## 1. Requirements

```bash
pip install torch numpy scipy matplotlib
```

Python 3.9+ recommended. No GPU is required — everything trains fast enough on CPU (a few minutes to
~30 min per run). All scripts are run from the **repository root**.

## 2. `PIRL/` — Prius-style HEV

### 2.1 Standalone demo
The simplest entry point: trains one PIRL agent and evaluates it against a rule-based baseline on
several standard driving cycles.

```bash
python PIRL/pirl_hev.py
```
Outputs: `PIRL/pirl_actor.pt` (trained weights) and `PIRL/pirl_soc_nedc.png` (SOC trace plot).

### 2.2 Full benchmark: PIRL vs DDPG vs DP vs Rule
`PIRL/benchmark.py` trains multiple algorithms/seeds and compares them against a Dynamic Programming
(DP) global optimum.

```bash
# Train one run (repeat per algo/seed/gamma you want in the comparison)
python PIRL/benchmark.py train --algo pirl --episodes 30 --seed 1
python PIRL/benchmark.py train --algo pirl --episodes 30 --seed 1 --gamma 0.9
python PIRL/benchmark.py train --algo ddpg --episodes 30 --seed 1
python PIRL/benchmark.py train --algo ddpg --episodes 100 --seed 1 --gamma 0.9

# Run DP + evaluate every trained run on 10 held-out driving cycles (nominal + plant-mismatch)
python PIRL/benchmark.py evaluate

# Build tables + figures into PIRL/results/
python PIRL/report.py
```
`--algo` accepts `pirl` or `ddpg`. Trained checkpoints/logs land in `PIRL/runs/<algo>_e<episodes>[_g<gamma>]_s<seed>.{pt,json}`.
`report.py` auto-selects the best `gamma` per algo based on the **training** cycle only (never the test
cycles) before building `PIRL/results/tables.md` and the PNG figures.

## 3. `PIRL_excavator/` — Parallel Hybrid Excavator

Engine + EMG on the pump shaft (parallel hybrid), with a separate generator/motor path that recovers
boom-lowering hydraulic energy into the battery.

```bash
cd PIRL_excavator

# 1. Tune the classical baselines (Rule, A-ECMS) on the validation cycle
python bench.py tune-classic

# 2. Train each algorithm (repeat per seed; algos: pirl, pirl_nofeat, pirl_noconstr,
#    pirl_nophyscritic, ddpg, td3, sac)
python bench.py train --algo pirl --gamma 0.99 --seed 1
python bench.py train --algo pirl --gamma 0.99 --seed 2
python bench.py train --algo pirl --gamma 0.99 --seed 3
python bench.py train --algo sac  --gamma 0.99 --seed 1
# ... repeat for td3, ddpg, and the pirl_no* ablations

# 3. Pick the champion run(s) per algo -> results/selected.json
python report.py select

# 4. Evaluate DP + every selected policy on all 10 held-out jobs, both plants
#    (nominal and mismatch = worn engine/pump + aged battery + hot ambient)
for plant in nominal mismatch; do
  for job in train_mixed heavy_dig truck_load_180 trenching grading pipe_lifting \
             breaker dig_with_waiting eco_dig dig_and_travel rock_excavation; do
    python bench.py evaluate --plant "$plant" --job "$job"
  done
done
python bench.py merge

# 5. Build tables + figures into results/
python report.py
```
Each `train`/`evaluate` call is independent, so steps 2 and 4 are easy to parallelize, e.g.:
```bash
printf 'nominal\nmismatch\n' | while read -r p; do
  printf 'train_mixed\nheavy_dig\ntruck_load_180\ntrenching\ngrading\npipe_lifting\nbreaker\ndig_with_waiting\neco_dig\ndig_and_travel\nrock_excavation\n' | \
    xargs -P 4 -I{} python bench.py evaluate --plant "$p" --job {}
done
```
`results/selected.json` must exist (step 3) before step 4, and `results/classic_params.json` must
exist (step 1) before step 4.

## 4. `PIRL_excavator_ps/` — Power-Split Hybrid Excavator

Planetary-gear power-split drivetrain: **carrier = ICE**, **sun = pump output shaft**, **ring = EMG**.
Two clutch-selected modes: **M1 (lifting/working)**, CM engaged / C2 open, ICE+EMG split power through
the planetary gear; **M2 (boom lowering)**, main valve closed, CM open / C2 engaged, the hydraulic
motor spins the EMG as a generator to charge the battery.

Adds two extra policies on top of plain PIRL:
- **PIRL-P**: at run time, picks the action that maximizes `Q_phys` over a 9×9 grid of candidates
  (plus the actor's own proposal) instead of using the actor output directly.
- **PIRL-P+**: same as PIRL-P, and the critic's TD target is also the max over a grid (5×7) during
  training, not just the actor's action.

```bash
cd PIRL_excavator_ps

# 1. Tune Rule / A-ECMS on the validation cycle
python bench.py tune-classic

# 2. Train (algos: pirl, pirlp [= trains the critic used by PIRL-P+], ddpg, td3, sac)
python bench.py train --algo pirl  --gamma 0.99 --seed 1
python bench.py train --algo pirl  --gamma 0.99 --seed 2
python bench.py train --algo pirl  --gamma 0.99 --seed 3
python bench.py train --algo pirlp --gamma 0.99 --seed 1   # PIRL-P+'s critic
python bench.py train --algo pirlp --gamma 0.99 --seed 2
python bench.py train --algo pirlp --gamma 0.99 --seed 3
python bench.py train --algo sac   --gamma 0.99 --seed 1
# ... repeat for td3, ddpg

# 3. Write results/selected.json by hand (or adapt PIRL_excavator/report.py's `select()`):
#    {"PIRL": ["pirl_g0.99_s1", ...], "PIRL-P": ["pirl_g0.99_s1", ...],
#     "PIRL-P+": ["pirlp_g0.99_s1", ...], "SAC": [...], "TD3": [...], "DDPG": [...]}
#    (PIRL and PIRL-P share the same trained actor+critic; only the run-time action
#    selection differs — see agents.py's `plan` flag / `load_policy(tag, plan=...)`.)

# 4. Evaluate DP (on a relaxed/idealized plant — the DP lower bound) + every policy,
#    both plants, all 10 jobs
for plant in nominal mismatch; do
  for job in train_mixed heavy_dig truck_load_180 trenching grading pipe_lifting \
             breaker dig_with_waiting eco_dig dig_and_travel rock_excavation; do
    python bench.py evaluate --plant "$plant" --job "$job"
  done
done
python bench.py merge

# 5. Build tables + figures
python report.py
```

### Diagnostics / auxiliary scripts
```bash
# Check whether the PIRL actor's output has saturated (sigmoid stuck near 0/1 => dead gradient)
python diag_actor.py

# Quick sweep of anti-saturation hyperparameters (logit regularization, Q-normalization)
# over a few episodes, without a full training run:
python try_variant.py <seed> <logit_reg> <use_q_normalization: 0|1>
```

## 5. Where results already live

Training runs, checkpoints and evaluation results from this session are already committed:
- `PIRL/runs/`, `PIRL/results/`
- `PIRL_excavator/runs/`, `PIRL_excavator/results/`
- `PIRL_excavator_ps/runs/`, `PIRL_excavator_ps/results/`

so you can read `results/tables.md` and the PNGs in each `results/` folder directly without
re-running anything. Re-running `report.py` overwrites `results/tables.md` and the figures in place;
re-running `train`/`evaluate` overwrites the corresponding files under `runs/`/`results/`.

## 6. Estimated run times (4-core CPU, this session's measurements)

| Step | Approx. time |
|---|---|
| `PIRL/pirl_hev.py` | ~2 min |
| `PIRL/benchmark.py train` (1 run, 30 episodes) | ~4 min |
| `PIRL_excavator/bench.py train` (1 run, 30 episodes) | ~4–15 min depending on algo |
| `PIRL_excavator_ps/bench.py train --algo pirl` (1 run) | ~25 min |
| `PIRL_excavator_ps/bench.py train --algo pirlp` (1 run) | ~35 min |
| `bench.py evaluate` (1 job × 1 plant, excavator variants) | ~2–3.5 min (DP dominates) |
| `report.py` | seconds to ~1 min (regenerates all figures) |
