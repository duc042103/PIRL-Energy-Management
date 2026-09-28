# -*- coding: utf-8 -*-
"""
Benchmark PIRL vs DDPG vs DP (+ rule-based) tren 10 chu trinh lai chua thay khi train.

Tat ca thuat toan dung CUNG mo hinh xe (HEVPhysics), CUNG reward, CUNG chu trinh train (UDDS).
Khac biet duy nhat giua PIRL va DDPG la vat ly co duoc nhung vao mang hay khong.

  python PIRL/benchmark.py train --algo pirl  --episodes 30  --seed 0
  python PIRL/benchmark.py train --algo ddpg  --episodes 30  --seed 0
  python PIRL/benchmark.py evaluate            # DP + danh gia tat ca run da train
"""
import os
import sys
import json
import time
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pirl_hev import (HEVPhysics, PHYS, PIRLAgent, load_cycle, reward_phys,  # noqa: E402
                      rule_based, SOC_REF)

RUN_DIR = os.path.join(HERE, 'runs')
RES_DIR = os.path.join(HERE, 'results')
TRAIN_CYCLE = 'Standard_UDDS'
TEST_CYCLES = ['Standard_NEDC', 'FTP75-2', 'Standard_HWFET', 'Standard_US06_2',
               'Standard_LA92_2', 'Standard_JN1015', 'Standard_ChinaCity',
               'Standard_WVUCITY2', 'Standard_WVUINTER', 'Standard_WVUSUB']
LHV, RHO_FUEL = 42600.0, 745.0          # J/g, g/L (xang)
EQ_FACTOR = PHYS.f1 / 0.9               # g nhien lieu tuong duong cho 1 J dien lay tu pin


def plant_mismatch():
    """Xe 'thuc' khac mo hinh: cho them 10% tai, Cd +10%, pin lao hoa (R +30%, Q -10%)."""
    p = HEVPhysics()
    p.m *= 1.10
    p.Cd *= 1.10
    p.R *= 1.30
    p.Q *= 0.90
    return p


def _t(x):
    return torch.tensor([[float(x)]], dtype=torch.float64)


# =============================================================================
# Mo phong + chi so danh gia
# =============================================================================
def simulate(policy, v, acc, soc0=SOC_REF, plant=PHYS, on_step=None):
    """Chay 1 chu trinh. Bo giam sat (supervisor) chieu P_eng vao mien kha thi cua xe
    - neu action cua agent bi sua thi dem la 1 lan 'can thiep' (vi pham rang buoc)."""
    soc = soc0
    tr = {k: [] for k in ('soc', 'P_eng', 'P_batt', 'P_req', 'I', 'Voc', 'fuel', 'interv', 'interv_batt', 'unmet')}
    for k in range(len(v) - 1):
        s = np.array([soc, v[k], acc[k]], dtype=np.float32)
        P_cmd = float(policy(s))
        with torch.no_grad():
            S, V, A = _t(soc), _t(v[k]), _t(acc[k])
            lo, hi = plant.engine_bounds(S, V, A)
            P = min(max(P_cmd, lo.item()), hi.item())
            d = plant.step_detail(S, V, A, _t(P))
        soc_next = d['soc_next'].item()
        P_req, P_batt = d['P_req'].item(), d['P_batt'].item()
        tr['soc'].append(soc_next); tr['P_eng'].append(P); tr['P_batt'].append(P_batt)
        tr['P_req'].append(P_req); tr['I'].append(d['I'].item()); tr['Voc'].append(d['Voc'].item())
        tr['fuel'].append(d['fuel'].item())
        tr['interv'].append(abs(P - P_cmd) > 100.0)
        tr['interv_batt'].append(abs(P - P_cmd) > 100.0 and P_req > 0)   # dung gioi han pin/SOC
        tr['unmet'].append(max(0.0, P_req - P - P_batt) if P_req > 0 else 0.0)
        if on_step is not None:
            r = reward_phys(soc_next, d['fuel'].item())
            s_next = np.array([soc_next, v[k + 1], acc[k + 1]], dtype=np.float32)
            on_step(s, P_cmd, r, s_next)
        soc = soc_next
    return {k: np.asarray(x) for k, x in tr.items()}


def metrics(tr, v, plant=PHYS):
    dist_km = float(np.sum(v[:-1])) / 1000.0
    fuel = float(tr['fuel'].sum())
    E_batt = float(np.sum(tr['Voc'] * tr['I']))            # J hoa nang lay tu pin (net)
    fuel_eq = fuel + E_batt * EQ_FACTOR
    I = tr['I']
    on = tr['P_eng'] > 1500
    return {
        # ---- nang luong ----
        'fuel_g': fuel,
        'fuel_L100': fuel / RHO_FUEL / dist_km * 100,
        'fuel_eq_L100': fuel_eq / RHO_FUEL / dist_km * 100,    # da quy doi chenh lech SOC
        'energy_kWh100': (fuel * LHV + E_batt) / 3.6e6 / dist_km * 100,
        'eng_eff': float(tr['P_eng'][on].sum() / max(tr['fuel'][on].sum() * LHV, 1e-9)),
        'eng_starts': int(np.sum(np.diff(on.astype(int)) == 1)),
        # ---- pin ----
        'soc_end': float(tr['soc'][-1]),
        'soc_dev_end': float(abs(tr['soc'][-1] - SOC_REF)),
        'soc_rms': float(np.sqrt(np.mean((tr['soc'] - SOC_REF) ** 2))),
        'soc_min': float(tr['soc'].min()), 'soc_max': float(tr['soc'].max()),
        'throughput_Ah': float(np.abs(I).sum() / 3600),
        'ohmic_loss_kJ': float(np.sum(I ** 2) * plant.R / 1000),
        'I_rms': float(np.sqrt(np.mean(I ** 2))),
        'I_peak': float(np.abs(I).max()),
        # ---- an toan ----
        'interventions': int(tr['interv'].sum()),
        'interv_batt': int(tr['interv_batt'].sum()),
        'unmet_kJ': float(tr['unmet'].sum() / 1000),
    }


# =============================================================================
# DP - toi uu toan cuc (biet truoc toan bo chu trinh) -> can duoi cua nhien lieu
# =============================================================================
class DPPolicy:
    def __init__(self, v, acc, plant=PHYS, n_soc=801, n_u=101, w_term=1e6):
        self.v, self.acc, self.plant, self.k = v, acc, plant, 0
        self.grid = torch.linspace(plant.soc_min, plant.soc_max, n_soc, dtype=torch.float64)
        self.u = torch.linspace(0, 1, n_u, dtype=torch.float64)[None, :]
        N = len(v) - 1
        self.J = [None] * (N + 1)
        self.J[N] = w_term * (self.grid - SOC_REF) ** 2        # rang buoc giu SOC cuoi = 0.6
        with torch.no_grad():
            for k in range(N - 1, -1, -1):
                Q = self._q(self.grid[:, None], k)
                self.J[k] = Q.min(dim=1).values

    def _interp(self, J, soc):
        n = len(self.grid)
        x = (soc - self.grid[0]) / (self.grid[1] - self.grid[0])
        i0 = torch.clamp(x.floor().long(), 0, n - 2)
        w = torch.clamp(x - i0, 0, 1)
        return J[i0] * (1 - w) + J[i0 + 1] * w

    def _q(self, S, k):
        V = torch.full_like(S, float(self.v[k]))
        A = torch.full_like(S, float(self.acc[k]))
        lo, hi = self.plant.engine_bounds(S, V, A)
        P = lo + self.u * (hi - lo)
        soc_next, fuel = self.plant.step(S.expand_as(P), V.expand_as(P), A.expand_as(P), P)
        return fuel + self._interp(self.J[k + 1], soc_next)

    def __call__(self, s):
        with torch.no_grad():
            S = torch.tensor([[float(s[0])]], dtype=torch.float64)
            Q = self._q(S, self.k)
            lo, hi = self.plant.engine_bounds(S, S * 0 + float(self.v[self.k]),
                                              S * 0 + float(self.acc[self.k]))
            u = self.u[0, Q[0].argmin()]
            P = (lo + u * (hi - lo)).item()
        self.k += 1
        return P


# =============================================================================
# DDPG baseline - giong DDPG_Prius.py cua repo (mang 200-100-50, action = sigmoid * 56 kW)
# nhung dung CUNG reward, CUNG chuan hoa state, CUNG hyper-parameter voi PIRL
# =============================================================================
def _norm(s):
    return torch.cat([(s[:, 0:1] - SOC_REF) * 10, s[:, 1:2] / 30, s[:, 2:3] / 2], dim=1)


def _net(n_in, out_act=None):
    layers = [nn.Linear(n_in, 200), nn.ReLU(), nn.Linear(200, 100), nn.ReLU(),
              nn.Linear(100, 50), nn.ReLU(), nn.Linear(50, 1)]
    return nn.Sequential(*(layers + ([out_act] if out_act else [])))


class DDPGAgent:
    def __init__(self, gamma=0.99, lr_a=1e-3, lr_c=1e-3, tau=0.005, buffer=100_000):
        import copy
        self.actor, self.critic = _net(3, nn.Sigmoid()), _net(4)
        self.actor_t, self.critic_t = copy.deepcopy(self.actor), copy.deepcopy(self.critic)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=lr_a)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=lr_c)
        self.gamma, self.tau = gamma, tau
        self.buf = np.zeros((buffer, 8), dtype=np.float32)       # s(3) u(1) r(1) s'(3)
        self.n, self.cap = 0, buffer

    @torch.no_grad()
    def act(self, s, noise_std=0.0):
        u = self.actor(_norm(torch.as_tensor(s)[None])).item()
        u = float(np.clip(u + noise_std * np.random.randn(), 0.0, 1.0))
        return u * PHYS.P_eng_max

    def store(self, s, P_cmd, r, s_next):
        self.buf[self.n % self.cap] = np.concatenate([s, [P_cmd / PHYS.P_eng_max, r], s_next])
        self.n += 1

    def update(self, batch=256):
        idx = np.random.randint(0, min(self.n, self.cap), batch)
        b = torch.as_tensor(self.buf[idx])
        s, u, r, s2 = b[:, :3], b[:, 3:4], b[:, 4:5], b[:, 5:]
        with torch.no_grad():
            y = r + self.gamma * self.critic_t(torch.cat([_norm(s2), self.actor_t(_norm(s2))], 1))
        loss_c = ((self.critic(torch.cat([_norm(s), u], 1)) - y) ** 2).mean()
        self.opt_c.zero_grad(); loss_c.backward(); self.opt_c.step()
        loss_a = -self.critic(torch.cat([_norm(s), self.actor(_norm(s))], 1)).mean()
        self.opt_a.zero_grad(); loss_a.backward(); self.opt_a.step()
        for net, tgt in ((self.actor, self.actor_t), (self.critic, self.critic_t)):
            for p, pt in zip(net.parameters(), tgt.parameters()):
                pt.data.mul_(1 - self.tau).add_(self.tau * p.data)


# =============================================================================
# Train (cung ngan sach cho moi thuat toan)
# =============================================================================
def train(algo, episodes, seed, gamma=0.99, tag=None):
    torch.manual_seed(seed); np.random.seed(seed); torch.set_num_threads(1)
    v, acc = load_cycle(TRAIN_CYCLE)
    agent = PIRLAgent(gamma=gamma) if algo == 'pirl' else DDPGAgent(gamma=gamma)

    if algo == 'pirl':
        def on_step(s, P_cmd, r, s_next):
            agent.store(s, s_next[1:])
            if agent.n >= 1000:
                agent.update()
    else:
        def on_step(s, P_cmd, r, s_next):
            agent.store(s, P_cmd, r, s_next)
            if agent.n >= 1000:
                agent.update()

    log, t0 = [], time.time()
    for ep in range(episodes):
        frac = ep / episodes
        noise = max(0.05, 1.0 * (1 - frac)) if algo == 'pirl' else max(0.02, 0.3 * (1 - frac))
        soc0 = np.random.uniform(0.5, 0.7)
        tr = simulate(lambda s: agent.act(s, noise), v, acc, soc0, on_step=on_step)
        ev = metrics(simulate(agent.act, v, acc), v)          # danh gia khong noise
        log.append({'ep': ep, 'time_s': time.time() - t0,
                    'train_interventions': int(tr['interv'].sum()),
                    'train_soc_min': float(tr['soc'].min()), 'train_soc_max': float(tr['soc'].max()),
                    'eval_fuel_eq_L100': ev['fuel_eq_L100'], 'eval_soc_end': ev['soc_end'],
                    'eval_interventions': ev['interventions']})
        print(f"[{algo} s{seed}] ep {ep:3d} fuel_eq {ev['fuel_eq_L100']:.3f} L/100km "
              f"SOC_end {ev['soc_end']:.3f} interv(train) {log[-1]['train_interventions']}", flush=True)

    tag = tag or f'{algo}_e{episodes}' + (f'_g{gamma}' if gamma != 0.99 else '') + f'_s{seed}'
    os.makedirs(RUN_DIR, exist_ok=True)
    torch.save(agent.actor.state_dict(), os.path.join(RUN_DIR, tag + '.pt'))
    with open(os.path.join(RUN_DIR, tag + '.json'), 'w') as f:
        json.dump({'algo': algo, 'episodes': episodes, 'seed': seed, 'gamma': gamma,
                   'train_time_s': time.time() - t0, 'log': log}, f, indent=1)


def load_policy(tag):
    meta = json.load(open(os.path.join(RUN_DIR, tag + '.json')))
    if meta['algo'] == 'pirl':
        agent = PIRLAgent(gamma=meta['gamma'])
    else:
        agent = DDPGAgent(gamma=meta['gamma'])
    agent.actor.load_state_dict(torch.load(os.path.join(RUN_DIR, tag + '.pt')))
    return agent.act, meta


# =============================================================================
# Evaluate
# =============================================================================
def evaluate():
    torch.set_num_threads(4)
    groups = {}
    for p in sorted(glob.glob(os.path.join(RUN_DIR, '*.json'))):
        tag = os.path.basename(p)[:-5]
        meta = json.load(open(p))
        name = f"{meta['algo'].upper()}-{meta['episodes']}ep"
        if meta['gamma'] != 0.99:
            name += f"-g{meta['gamma']}"
        groups.setdefault(name, []).append(tag)

    results = {}          # results[plant][cycle][method] = list of metric dicts (1 per seed)
    traces = {}
    for plant_name, plant in (('nominal', PHYS), ('mismatch', plant_mismatch())):
        results[plant_name] = {}
        for cyc in [TRAIN_CYCLE] + TEST_CYCLES:
            v, acc = load_cycle(cyc)
            R = results[plant_name][cyc] = {}
            t0 = time.time()
            tr = simulate(DPPolicy(v, acc, plant), v, acc, plant=plant)
            R['DP'] = [metrics(tr, v, plant)]
            traces[(plant_name, cyc, 'DP')] = tr['soc']
            tr = simulate(rule_based, v, acc, plant=plant)
            R['Rule'] = [metrics(tr, v, plant)]
            traces[(plant_name, cyc, 'Rule')] = tr['soc']
            for name, tags in groups.items():
                R[name] = []
                for i, tag in enumerate(tags):
                    pol, _ = load_policy(tag)
                    tr = simulate(pol, v, acc, plant=plant)
                    R[name].append(metrics(tr, v, plant))
                    if i == 0:
                        traces[(plant_name, cyc, name)] = tr['soc']
            print(f'{plant_name:8s} {cyc:18s} done ({time.time() - t0:.0f}s) | ' +
                  ' '.join(f"{m}={np.mean([x['fuel_eq_L100'] for x in R[m]]):.3f}" for m in R),
                  flush=True)

    os.makedirs(RES_DIR, exist_ok=True)
    with open(os.path.join(RES_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=1)
    np.savez_compressed(os.path.join(RES_DIR, 'soc_traces.npz'),
                        **{'|'.join(k): x for k, x in traces.items()})


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['train', 'evaluate'])
    ap.add_argument('--algo', choices=['pirl', 'ddpg'])
    ap.add_argument('--episodes', type=int, default=30)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--gamma', type=float, default=0.99)
    ap.add_argument('--tag')
    a = ap.parse_args()
    if a.cmd == 'train':
        train(a.algo, a.episodes, a.seed, a.gamma, a.tag)
    else:
        evaluate()
