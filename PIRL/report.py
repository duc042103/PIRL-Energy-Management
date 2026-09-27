# -*- coding: utf-8 -*-
"""Tao bang (RESULTS.md) va hinh tu PIRL/results/results.json + PIRL/runs/*.json.

  python PIRL/report.py
"""
import os
import sys
import json
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from benchmark import RUN_DIR, RES_DIR, TRAIN_CYCLE, TEST_CYCLES  # noqa: E402

COLORS = {'DP': '#52514e', 'PIRL-30ep': '#2a78d6', 'DDPG-30ep': '#eb6834',
          'DDPG-100ep': '#1baf7a', 'Rule': '#b5b3ab'}
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID,
                     'grid.linewidth': 0.6, 'legend.frameon': False, 'figure.facecolor': '#fcfcfb',
                     'axes.facecolor': '#fcfcfb'})

SHORT = {c: c.replace('Standard_', '').replace('_2', '').replace('-2', '') for c in [TRAIN_CYCLE] + TEST_CYCLES}

# (key, label, unit, lower_is_better, fmt)
METRICS = [
    ('fuel_eq_L100', 'Nhiên liệu quy đổi SOC', 'L/100km', True, '{:.3f}'),
    ('gap_dp', 'Chênh lệch so với DP', '%', True, '{:+.2f}'),
    ('energy_kWh100', 'Tổng năng lượng (nhiên liệu + pin)', 'kWh/100km', True, '{:.2f}'),
    ('eng_eff', 'Hiệu suất TB động cơ', '%', False, '{:.2f}'),
    ('eng_starts', 'Số lần khởi động máy', '', True, '{:.1f}'),
    ('soc_dev_end', '|SOC cuối − 0.6|', '', True, '{:.4f}'),
    ('soc_rms', 'RMS(SOC − 0.6)', '', True, '{:.4f}'),
    ('throughput_Ah', 'Lưu lượng Ah qua pin (lão hoá)', 'Ah', True, '{:.2f}'),
    ('ohmic_loss_kJ', 'Tổn hao nhiệt I²R trong pin', 'kJ', True, '{:.1f}'),
    ('I_rms', 'Dòng pin RMS', 'A', True, '{:.1f}'),
    ('I_peak', 'Dòng pin đỉnh', 'A', True, '{:.1f}'),
    ('interventions', 'Số bước action bị supervisor sửa (tổng)', '', True, '{:.1f}'),
    ('interv_batt', '… trong đó vi phạm giới hạn pin/SOC', '', True, '{:.1f}'),
]


def load():
    res = json.load(open(os.path.join(RES_DIR, 'results.json')))
    for plant in res.values():
        for cyc, R in plant.items():
            dp = R['DP'][0]['fuel_eq_L100']
            for runs in R.values():
                for m in runs:
                    m['gap_dp'] = 100 * (m['fuel_eq_L100'] / dp - 1)
                    m['eng_eff'] = m['eng_eff'] * 100 if m['eng_eff'] < 1.5 else m['eng_eff']
    return res


def methods_of(res):
    order = ['DP', 'PIRL-30ep', 'DDPG-30ep', 'DDPG-100ep', 'Rule']
    have = list(next(iter(res['nominal'].values())).keys())
    return [m for m in order if m in have] + [m for m in have if m not in order]


def agg(R, method, key):
    """mean, std qua cac seed."""
    x = np.array([m[key] for m in R[method]], dtype=float)
    return x.mean(), x.std()


def fmt_ms(mean, std, f, show_std):
    return f.format(mean) + (f' ± {f.format(std).lstrip("+")}' if show_std and std > 0 else '')


def per_cycle_table(res, plant, key, f):
    methods = methods_of(res)
    lines = ['| Chu trình | ' + ' | '.join(methods) + ' | PIRL tốt hơn DDPG? |',
             '|' + '---|' * (len(methods) + 2)]
    wins = 0
    for cyc in [TRAIN_CYCLE] + TEST_CYCLES:
        R = res[plant][cyc]
        vals = {m: agg(R, m, key) for m in methods}
        rl = [m for m in methods if m not in ('DP', 'Rule')]
        best = min(rl, key=lambda m: vals[m][0])
        ddpg_best = min((vals[m][0] for m in rl if m.startswith('DDPG')), default=np.inf)
        win = vals['PIRL-30ep'][0] < ddpg_best
        wins += win and cyc != TRAIN_CYCLE
        cells = []
        for m in methods:
            c = fmt_ms(*vals[m], f, m not in ('DP', 'Rule'))
            cells.append(f'**{c}**' if m == best else c)
        name = SHORT[cyc] + (' *(train)*' if cyc == TRAIN_CYCLE else '')
        lines.append(f'| {name} | ' + ' | '.join(cells) + f" | {'✅' if win else '❌'} |")
    return '\n'.join(lines), wins


def summary_table(res, plant):
    """Trung binh tren 10 chu trinh test (seed-mean truoc, roi cycle-mean)."""
    methods = methods_of(res)
    lines = ['| # | Chỉ số (TB 10 chu trình test) | Đơn vị | ' + ' | '.join(methods) + ' |',
             '|' + '---|' * (len(methods) + 3)]
    for i, (key, label, unit, low, f) in enumerate(METRICS, 1):
        vals = {m: np.mean([agg(res[plant][c], m, key)[0] for c in TEST_CYCLES]) for m in methods}
        rl = [m for m in methods if m not in ('DP', 'Rule')]
        best = (min if low else max)(rl, key=lambda m: vals[m])
        cells = [f'**{f.format(vals[m])}**' if m == best else f.format(vals[m]) for m in methods]
        lines.append(f'| {i} | {label} | {unit} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def training_table():
    groups = {}
    for p in sorted(glob.glob(os.path.join(RUN_DIR, '*.json'))):
        meta = json.load(open(p))
        name = f"{meta['algo'].upper()}-{meta['episodes']}ep"
        if meta['algo'] == 'ddpg' and meta['gamma'] != 0.99:
            name += f"-g{meta['gamma']}"
        groups.setdefault(name, []).append(meta)
    lines = ['| Thuật toán | Số seed | Thời gian train (phút) | Vi phạm ràng buộc khi train (tổng) '
             '| SOC min / max khi train | Episode đầu tiên đạt ≤ 1.02 × kết quả cuối |',
             '|---|---|---|---|---|---|']
    for name, metas in groups.items():
        t = np.mean([m['train_time_s'] for m in metas]) / 60
        viol = np.mean([sum(e['train_interventions'] for e in m['log']) for m in metas])
        smin = min(min(e['train_soc_min'] for e in m['log']) for m in metas)
        smax = max(max(e['train_soc_max'] for e in m['log']) for m in metas)
        conv = []
        for m in metas:
            f = np.array([e['eval_fuel_eq_L100'] for e in m['log']])
            conv.append(int(np.argmax(f <= 1.02 * f[-1])))
        lines.append(f'| {name} | {len(metas)} | {t:.1f} | {viol:.0f} | {smin:.3f} / {smax:.3f} '
                     f'| {np.mean(conv):.1f} |')
    return '\n'.join(lines), groups


# ------------------------------------------------------------------ figures
def fig_gap(res, path):
    methods = [m for m in methods_of(res) if m != 'DP']
    cycles = TEST_CYCLES
    fig, ax = plt.subplots(figsize=(10, 3.8))
    w = 0.8 / len(methods)
    x = np.arange(len(cycles))
    for i, m in enumerate(methods):
        mean = [agg(res['nominal'][c], m, 'gap_dp')[0] for c in cycles]
        ax.bar(x + (i - (len(methods) - 1) / 2) * w, mean, w * 0.9, color=COLORS.get(m), label=m,
               edgecolor='#fcfcfb', linewidth=1)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(x, [SHORT[c] for c in cycles])
    ax.set_ylabel('Nhiên liệu vượt DP (%)')
    ax.set_title('Khoảng cách tới tối ưu toàn cục DP trên 10 chu trình test (thấp hơn = tốt hơn)',
                 loc='left', color=INK)
    ax.legend(ncol=len(methods), loc='upper left')
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_learning(groups, path):
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
    for name, metas in groups.items():
        if '-g' in name:
            continue
        f = np.array([[e['eval_fuel_eq_L100'] for e in m['log']] for m in metas])
        iv = np.array([[e['train_interventions'] for e in m['log']] for m in metas])
        ep = np.arange(1, f.shape[1] + 1)
        c = COLORS.get(name)
        axs[0].plot(ep, f.mean(0), color=c, lw=2, label=name)
        axs[0].fill_between(ep, f.min(0), f.max(0), color=c, alpha=0.15, lw=0)
        axs[1].plot(ep, iv.mean(0), color=c, lw=2, label=name)
    axs[0].set_ylim(top=min(axs[0].get_ylim()[1], 8))
    axs[0].set_title('Nhiên liệu quy đổi trên UDDS sau mỗi episode', loc='left', color=INK)
    axs[0].set_xlabel('Episode'); axs[0].set_ylabel('L/100km'); axs[0].legend()
    axs[1].set_title('Số bước vi phạm ràng buộc trong mỗi episode train', loc='left', color=INK)
    axs[1].set_xlabel('Episode'); axs[1].set_ylabel('bước / episode')
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_soc(res, path):
    tr = np.load(os.path.join(RES_DIR, 'soc_traces.npz'))
    cycles = ['Standard_NEDC', 'FTP75-2', 'Standard_US06_2']
    methods = [m for m in ['DP', 'PIRL-30ep', 'DDPG-100ep', 'DDPG-30ep'] if f'nominal|{cycles[0]}|{m}' in tr]
    fig, axs = plt.subplots(len(cycles), 1, figsize=(10, 7), sharex=False)
    for ax, c in zip(axs, cycles):
        for m in methods:
            ax.plot(tr[f'nominal|{c}|{m}'], color=COLORS[m], lw=1.6 if m != 'DP' else 2.2, label=m)
        ax.axhline(0.6, color=MUTED, lw=0.8, ls='--')
        ax.set_ylabel('SOC'); ax.set_title(SHORT[c], loc='left', color=INK)
    axs[0].legend(ncol=len(methods), loc='upper left'); axs[-1].set_xlabel('Thời gian (s)')
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_battery(res, path):
    methods = methods_of(res)
    keys = [('throughput_Ah', 'Lưu lượng Ah'), ('ohmic_loss_kJ', 'Tổn hao I²R (kJ)'),
            ('I_rms', 'Dòng RMS (A)'), ('soc_rms', 'RMS(SOC − 0.6)')]
    fig, axs = plt.subplots(1, 4, figsize=(11, 3))
    for ax, (k, lab) in zip(axs, keys):
        vals = [np.mean([agg(res['nominal'][c], m, k)[0] for c in TEST_CYCLES]) for m in methods]
        ax.bar(range(len(methods)), vals, color=[COLORS.get(m) for m in methods], width=0.7)
        ax.set_xticks(range(len(methods)), methods, rotation=35, ha='right')
        ax.set_title(lab, loc='left', color=INK); ax.grid(axis='x', visible=False)
    fig.suptitle('Chỉ số pin, trung bình 10 chu trình test (thấp hơn = pin ít chịu tải hơn)',
                 x=0.01, ha='left', color=INK)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    res = load()
    tt, groups = training_table()
    t_fuel, wins = per_cycle_table(res, 'nominal', 'fuel_eq_L100', '{:.3f}')
    t_gap, _ = per_cycle_table(res, 'nominal', 'gap_dp', '{:+.2f}')
    t_mis, wins_mis = per_cycle_table(res, 'mismatch', 'fuel_eq_L100', '{:.3f}')
    t_abl, _ = per_cycle_table(res, 'nominal', 'throughput_Ah', '{:.2f}')
    fig_gap(res, os.path.join(RES_DIR, 'fig_gap_to_dp.png'))
    fig_learning(groups, os.path.join(RES_DIR, 'fig_learning.png'))
    fig_soc(res, os.path.join(RES_DIR, 'fig_soc.png'))
    fig_battery(res, os.path.join(RES_DIR, 'fig_battery.png'))
    out = {'summary_nominal': summary_table(res, 'nominal'),
           'summary_mismatch': summary_table(res, 'mismatch'),
           'per_cycle_fuel': t_fuel, 'per_cycle_gap': t_gap, 'per_cycle_mismatch': t_mis,
           'per_cycle_throughput': t_abl, 'training': tt,
           'wins': wins, 'wins_mismatch': wins_mis}
    with open(os.path.join(RES_DIR, 'tables.md'), 'w') as f:
        for k, v in out.items():
            f.write(f'### {k}\n\n{v}\n\n')
    print(open(os.path.join(RES_DIR, 'tables.md')).read())


if __name__ == '__main__':
    main()
