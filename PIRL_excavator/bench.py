# -*- coding: utf-8 -*-
"""
Benchmark EMS may xuc hybrid.

  python PIRL_excavator/bench.py train --algo pirl --gamma 0.95 --seed 0
  python PIRL_excavator/bench.py tune-classic        # tim tham so Rule / A-ECMS tren chu trinh validation
  python PIRL_excavator/bench.py evaluate            # DP + moi policy tren 10 cong viec test x 2 kich ban
"""
import os
import sys
import json
import time
import glob
import argparse
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from excavator_model import PHYS, plant_mismatch, reward, LHV, RHO_DIESEL      # noqa: E402
from duty_cycles import make_cycle, TEST_JOBS                                   # noqa: E402
import agents as A                                                              # noqa: E402

RUN_DIR, RES_DIR = os.path.join(HERE, 'runs'), os.path.join(HERE, 'results')
EPISODES = 30
VAL = ('train_mixed', 999)
EQ = 1.0 / (0.40 * LHV)             # g diesel / J dien hoa nang lay tu pin
EQ_CONS = 1.0 / (0.30 * 0.85 * LHV)  # he so bat loi (nap lai pin kem hieu qua)
DIESEL_USD_L, BATT_USD_AH = 1.3, 0.01225
ALGOS = {
    'pirl': lambda g: A.PIRLAgent(g),
    'pirl_nofeat': lambda g: A.PIRLAgent(g, feat=False),
    'pirl_noconstr': lambda g: A.PIRLAgent(g, constr=False),
    'pirl_nophyscritic': lambda g: A.PIRLAgent(g, phys_critic=False),
    'ddpg': lambda g: A.DDPGAgent(g),
    'td3': lambda g: A.TD3Agent(g),
    'sac': lambda g: A.SACAgent(g),
}


def simulate(policy, cyc, soc0=0.55, T0=38.0, plant=PHYS, on_step=None):
    """Moi truong + supervisor: chieu (n, P_eng) vao mien kha thi cua MAY THUC (plant)."""
    N = len(cyc['n'])
    exo = np.stack([cyc['n'], cyc['P_hyd'], cyc['P_boom'], cyc['w_hm']], 1)
    soc, Tb, n_cur = soc0, T0, float(exo[0][0])
    keys = ('soc', 'Tb', 'n', 'P_eng', 'fuel', 'age', 'I', 'V', 'R', 'P_rec', 'curtail', 'unmet',
            'interv', 'P_pump', 'r')
    tr = {k: np.zeros(N - 1) for k in keys}
    for k in range(N - 1):
        s = np.array([soc, Tb, n_cur, *exo[k]], dtype=np.float32)
        n_cmd, P_cmd = policy(s)
        with torch.no_grad():
            st = torch.as_tensor(s[None], dtype=torch.float64)
            n, P = A.project(st, torch.tensor([n_cmd], dtype=torch.float64),
                             torch.tensor([P_cmd], dtype=torch.float64), plant)
            d = plant.step_detail(st, n, P)
            r = reward(d, plant).item()
        n, P = n.item(), P.item()
        vals = dict(soc=d['soc'].item(), Tb=d['Tb'].item(), n=n, P_eng=P, fuel=d['fuel'].item(),
                    age=d['age_Ah'].item(), I=d['I'].item(), V=d['V'].item(), R=d['R'].item(),
                    P_rec=d['P_rec'].item(), curtail=d['curtail'].item(), unmet=d['unmet'].item(),
                    interv=abs(P - P_cmd) > 200.0 or abs(n - n_cmd) > 10.0,
                    P_pump=d['P_pump'].item(), r=r)
        for kk, vv in vals.items():
            tr[kk][k] = vv
        if on_step is not None:
            s2 = np.array([vals['soc'], vals['Tb'], n, *exo[k + 1]], dtype=np.float32)
            on_step(s, (n_cmd, P_cmd), r, s2)
        soc, Tb, n_cur = vals['soc'], vals['Tb'], n
    return tr


def metrics(tr, plant=PHYS):
    dt = plant.dt
    hours = len(tr['soc']) * dt / 3600
    fuel = tr['fuel'].sum()
    E_batt = float(np.sum(tr['V'] * tr['I']) * dt)                 # J hoa nang (net) lay tu pin
    fuel_eq = fuel + E_batt * EQ
    rec_av = tr['P_rec'].sum() * dt / 3.6e6
    rec_cut = tr['curtail'].sum() * dt / 3.6e6
    on = tr['P_eng'] > 1000
    age_h = tr['age'].sum() / hours
    return {
        # ---- nang luong
        'fuel_Lh': fuel / RHO_DIESEL / hours,
        'fuel_eq_Lh': fuel_eq / RHO_DIESEL / hours,
        'fuel_eq_cons_Lh': (fuel + E_batt * EQ_CONS) / RHO_DIESEL / hours,
        'bsfc_gkWh': float(fuel / max(tr['P_eng'][on].sum() * dt / 3.6e6, 1e-9)),
        'rec_avail_kWh': rec_av,
        'rec_used_kWh': rec_av - rec_cut,
        'rec_util_pct': 100 * (rec_av - rec_cut) / max(rec_av, 1e-9),
        'curtail_kWh': rec_cut,
        'unmet_kWh': tr['unmet'].sum() / 3.6e6,
        'n_mean': float(tr['n'].mean()),
        'cost_usd_h': fuel_eq / RHO_DIESEL / hours * DIESEL_USD_L + age_h * BATT_USD_AH,
        # ---- pin
        'soc_end': tr['soc'][-1],
        'soc_dev_end': abs(tr['soc'][-1] - plant.soc_ref),
        'soc_rms': float(np.sqrt(np.mean((tr['soc'] - plant.soc_ref) ** 2))),
        'soc_min': tr['soc'].min(), 'soc_max': tr['soc'].max(),
        'T_max': tr['Tb'].max(),
        'derate_pct': 100 * float(np.mean(tr['Tb'] > plant.T_derate)),
        'throughput_Ah_h': np.abs(tr['I']).sum() * dt / 3600 / hours,
        'aging_Ah_h': age_h,
        'batt_life_h': 200_000 / max(age_h, 1e-9),
        'ohmic_kJ': float(np.sum(tr['I'] ** 2 * tr['R']) * dt / 1e3),
        'I_rms': float(np.sqrt(np.mean(tr['I'] ** 2))),
        'I_peak': float(np.abs(tr['I']).max()),
        # ---- an toan
        'interventions': int(tr['interv'].sum()),
        'return': float(tr['r'].sum()),
    }


# =============================================================================
def train(algo, gamma, seed, episodes=EPISODES):
    torch.manual_seed(seed); np.random.seed(seed); torch.set_num_threads(1)
    agent = ALGOS[algo](gamma)
    val = make_cycle(*VAL)
    log, t0 = [], time.time()
    for ep in range(episodes):
        frac = ep / episodes
        cyc = make_cycle('train_mixed', seed=100 + ep + 1000 * seed)
        soc0, T0 = np.random.uniform(0.45, 0.65), np.random.uniform(33, 45)
        tr = simulate(lambda s: agent.act(s, True, frac), cyc, soc0, T0, on_step=agent.observe)
        ev = metrics(simulate(lambda s: agent.act(s), val))
        log.append({'ep': ep, 'time_s': time.time() - t0, 'train_interv': int(tr['interv'].sum()),
                    'train_T_max': float(tr['Tb'].max()), 'train_soc_min': float(tr['soc'].min()),
                    'train_soc_max': float(tr['soc'].max()), 'val_fuel_eq_Lh': ev['fuel_eq_Lh'],
                    'val_return': ev['return'], 'val_cost': ev['cost_usd_h']})
        print(f"[{algo} g{gamma} s{seed}] ep {ep:2d} val fuel_eq {ev['fuel_eq_Lh']:.3f} L/h "
              f"cost {ev['cost_usd_h']:.3f} $/h SOCend {ev['soc_end']:.3f} interv {log[-1]['train_interv']}",
              flush=True)
    os.makedirs(RUN_DIR, exist_ok=True)
    tag = f'{algo}_g{gamma}_s{seed}'
    torch.save(agent.__dict__.get('actor', getattr(agent, 'pi', None)).state_dict(),
               os.path.join(RUN_DIR, tag + '.pt'))
    json.dump({'algo': algo, 'gamma': gamma, 'seed': seed, 'train_time_s': time.time() - t0, 'log': log},
              open(os.path.join(RUN_DIR, tag + '.json'), 'w'), indent=1)


def load_policy(tag):
    meta = json.load(open(os.path.join(RUN_DIR, tag + '.json')))
    agent = ALGOS[meta['algo']](meta['gamma'])
    net = agent.actor if hasattr(agent, 'actor') else agent.pi
    net.load_state_dict(torch.load(os.path.join(RUN_DIR, tag + '.pt')))
    return (lambda s: agent.act(s)), meta


def tune_classic():
    """Luoi tham so cho Rule va A-ECMS tren chu trinh validation (giong cach chon gamma cho RL)."""
    val = make_cycle(*VAL)
    best = {}
    for tau_f in [4.0, 8.0, 15.0, 30.0]:
        for k in [1e5, 2e5, 4e5, 8e5]:
            m = metrics(simulate(A.RulePolicy(tau_f, k), val))
            if 'rule' not in best or m['return'] > best['rule'][1]:
                best['rule'] = ({'tau_f': tau_f, 'k_soc': k}, m['return'], m['fuel_eq_Lh'])
    for s0 in [0.6, 0.8, 1.0, 1.2, 1.5]:
        for kp in [0.0, 10.0, 30.0, 60.0]:
            m = metrics(simulate(A.ECMSPolicy(s0, kp), val))
            if 'ecms' not in best or m['return'] > best['ecms'][1]:
                best['ecms'] = ({'s0': s0, 'kp': kp}, m['return'], m['fuel_eq_Lh'])
    print(best)
    os.makedirs(RES_DIR, exist_ok=True)
    json.dump({k: v[0] for k, v in best.items()}, open(os.path.join(RES_DIR, 'classic_params.json'), 'w'))


def evaluate(tags=None):
    torch.set_num_threads(4)
    cp = json.load(open(os.path.join(RES_DIR, 'classic_params.json')))
    sel = json.load(open(os.path.join(RES_DIR, 'selected.json')))       # {name: [tags]}
    results, traces = {}, {}
    for plant_name, plant in (('nominal', PHYS), ('mismatch', plant_mismatch())):
        results[plant_name] = {}
        for job in ['train_mixed'] + TEST_JOBS:
            cyc = make_cycle(job, seed=999 if job == 'train_mixed' else 0)
            R = results[plant_name][job] = {}
            t0 = time.time()
            pols = {'DP': [A.DPPolicy(cyc, plant)],
                    'A-ECMS': [A.ECMSPolicy(**cp['ecms'], phys=plant)],
                    'Rule': [A.RulePolicy(**cp['rule'], phys=plant)]}
            for name, ts in sel.items():
                pols[name] = [load_policy(t)[0] for t in ts]
            for name, ps in pols.items():
                R[name] = []
                for i, p in enumerate(ps):
                    if hasattr(p, 'reset'):
                        p.reset()
                    tr = simulate(p, cyc, plant=plant)
                    R[name].append({k: float(v) for k, v in metrics(tr, plant).items()})
                    if i == 0 and job in ('trenching', 'pipe_lifting', 'heavy_dig'):
                        traces[f'{plant_name}|{job}|{name}'] = np.stack([tr['soc'], tr['Tb'], tr['P_eng'],
                                                                         tr['P_pump'], tr['curtail']])
            print(f'{plant_name:8s} {job:18s} ({time.time() - t0:.0f}s) ' +
                  ' '.join(f"{m}={np.mean([x['fuel_eq_Lh'] for x in R[m]]):.2f}" for m in R), flush=True)
    os.makedirs(RES_DIR, exist_ok=True)
    json.dump(results, open(os.path.join(RES_DIR, 'results.json'), 'w'), indent=1)
    np.savez_compressed(os.path.join(RES_DIR, 'traces.npz'), **traces)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['train', 'tune-classic', 'evaluate'])
    ap.add_argument('--algo', choices=list(ALGOS))
    ap.add_argument('--gamma', type=float, default=0.95)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--episodes', type=int, default=EPISODES)
    a = ap.parse_args()
    if a.cmd == 'train':
        train(a.algo, a.gamma, a.seed, a.episodes)
    elif a.cmd == 'tune-classic':
        tune_classic()
    else:
        evaluate()
