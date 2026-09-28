# -*- coding: utf-8 -*-
"""Bang + hinh cho benchmark may xuc POWER-SPLIT.

  python PIRL_excavator/report.py select   # chon gamma moi thuat toan (theo validation) -> selected.json
  python PIRL_excavator/report.py          # bang (results/tables.md) + hinh
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
from bench import RUN_DIR, RES_DIR, make_cycle        # noqa: E402
from duty_cycles import TEST_JOBS, DT                  # noqa: E402
from ps_model import RHO_DIESEL                        # noqa: E402

NAMES = {'pirl': 'PIRL', 'ddpg': 'DDPG', 'td3': 'TD3', 'sac': 'SAC',
         'pirl_nofeat': 'PIRL w/o feature', 'pirl_noconstr': 'PIRL w/o constraint',
         'pirl_nophyscritic': 'PIRL w/o phys-critic'}
MAIN = ['DP', 'PIRL-P', 'PIRL-P+', 'PIRL', 'SAC', 'TD3', 'DDPG', 'A-ECMS', 'Rule']
ABL = ['PIRL', 'PIRL w/o feature', 'PIRL w/o constraint', 'PIRL w/o phys-critic']
COLORS = {'DP': '#52514e', 'PIRL-P': '#2a78d6', 'PIRL-P+': '#4a3aa7', 'PIRL': '#8fb8ea', 'SAC': '#eb6834', 'TD3': '#1baf7a', 'DDPG': '#eda100',
          'A-ECMS': '#4a3aa7', 'Rule': '#b5b3ab', 'PIRL w/o feature': '#e87ba4',
          'PIRL w/o constraint': '#008300', 'PIRL w/o phys-critic': '#e34948'}
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID,
                     'grid.linewidth': 0.6, 'legend.frameon': False, 'figure.facecolor': '#fcfcfb',
                     'axes.facecolor': '#fcfcfb'})

# (key, label, lower_is_better, fmt)
METRICS = [
    ('sfc_work', 'Nhiên liệu quy đổi SOC / công thủy lực thực hiện (g/kWh)', True, '{:.1f}'),
    ('gap_dp', 'Chênh lệch so với DP, tính theo g/kWh công (%)', True, '{:+.1f}'),
    ('unmet_pct', 'Công việc không đáp ứng (% công thủy lực)', True, '{:.2f}'),
    ('fuel_eq_Lh', 'Nhiên liệu quy đổi SOC (L/h)', True, '{:.2f}'),
    ('fuel_eq_cons_Lh', 'Nhiên liệu, hệ số quy đổi bất lợi (L/h)', True, '{:.2f}'),
    ('cost_usd_h', 'Chi phí vận hành nhiên liệu + hao mòn pin ($/h)', True, '{:.2f}'),
    ('bsfc_gkWh', 'BSFC trung bình động cơ (g/kWh)', True, '{:.0f}'),
    ('n_mean', 'Tốc độ động cơ trung bình (rpm)', True, '{:.0f}'),
    ('rec_util_pct', 'Tỷ lệ tận dụng năng lượng hạ cần (%)', False, '{:.1f}'),
    ('curtail_kWh', 'Năng lượng hạ cần bị tiết lưu (kWh)', True, '{:.3f}'),
    ('soc_dev_end', '|SOC cuối − 0.55|', True, '{:.3f}'),
    ('soc_rms', 'RMS(SOC − 0.55)', True, '{:.3f}'),
    ('T_max', 'Nhiệt độ pin lớn nhất (°C)', True, '{:.1f}'),
    ('derate_pct', 'Thời gian pin bị giảm định mức, T > 40 °C (%)', True, '{:.1f}'),
    ('throughput_Ah_h', 'Lưu lượng Ah (Ah/h)', True, '{:.1f}'),
    ('aging_Ah_h', 'Hao mòn pin, Ah hiệu dụng (Ah/h)', True, '{:.1f}'),
    ('batt_life_h', 'Tuổi thọ pin ước tính (giờ máy)', False, '{:.0f}'),
    ('ohmic_kJ', 'Tổn hao I²R trong pin (kJ)', True, '{:.0f}'),
    ('I_rms', 'Dòng pin RMS (A)', True, '{:.1f}'),
    ('I_peak', 'Dòng pin đỉnh (A)', True, '{:.1f}'),
    ('interventions', 'Số bước action bị supervisor sửa', True, '{:.0f}'),
]


def runs():
    out = {}
    for p in sorted(glob.glob(os.path.join(RUN_DIR, '*.json'))):
        m = json.load(open(p))
        out[os.path.basename(p)[:-5]] = m
    return out


def score(meta):
    """Diem validation = return trung binh 5 episode cuoi (cang cao cang tot)."""
    return float(np.mean([e['val_return'] for e in meta['log'][-5:]]))


def select():
    """Chon gamma cho moi thuat toan theo seed 0 tren chu trinh validation (khong nhin test).
    Seed 0 chi dung de tinh chinh; ket qua bao cao dung seed 1-3."""
    R = runs()
    best = {}
    for tag, m in R.items():
        if m['seed'] != 0:
            continue
        if m['algo'] not in best or score(m) > best[m['algo']][1]:
            best[m['algo']] = (m['gamma'], score(m))
    print({k: v for k, v in best.items()})
    json.dump({a: g for a, (g, _) in best.items()}, open(os.path.join(RES_DIR, 'gamma.json'), 'w'))
    sel = {}
    for a, (g, _) in best.items():
        tags = sorted(t for t, m in R.items() if m['algo'] == a and m['gamma'] == g and m['seed'] != 0)
        sel[NAMES[a]] = tags
    for a in ('pirl_nofeat', 'pirl_noconstr', 'pirl_nophyscritic'):
        tags = sorted(t for t, m in R.items() if m['algo'] == a)
        if tags:
            sel[NAMES[a]] = tags
    json.dump(sel, open(os.path.join(RES_DIR, 'selected.json'), 'w'), indent=1)
    print(sel)


def load():
    res = json.load(open(os.path.join(RES_DIR, 'results.json')))
    for plant in res.values():
        for job, R in plant.items():
            cyc = make_cycle(job, seed=999 if job == 'train_mixed' else 0)
            demand = cyc['P_hyd'][:-1].sum() * DT / 3.6e6                 # kWh thuy luc yeu cau
            hours = (len(cyc['n']) - 1) * DT / 3600
            for rr in R.values():
                for m in rr:
                    done = demand - m['unmet_kWh']
                    m['unmet_pct'] = 100 * m['unmet_kWh'] / demand
                    m['sfc_work'] = m['fuel_eq_Lh'] * RHO_DIESEL * hours / done   # g diesel / kWh cong thuc hien
            dp = R['DP'][0]['sfc_work']
            for rr in R.values():
                for m in rr:
                    m['gap_dp'] = 100 * (m['sfc_work'] / dp - 1)
    return res


def agg(R, m, k):
    x = np.array([r[k] for r in R[m]], dtype=float)
    return x.mean(), x.std()


def summary(res, plant, methods):
    methods = [m for m in methods if m in res[plant][TEST_JOBS[0]]]
    L = ['| # | Chỉ số (TB 10 công việc test) | ' + ' | '.join(methods) + ' |', '|' + '---|' * (len(methods) + 2)]
    for i, (k, lab, low, f) in enumerate(METRICS, 1):
        v = {m: np.mean([agg(res[plant][j], m, k)[0] for j in TEST_JOBS]) for m in methods}
        cand = [m for m in methods if m != 'DP']
        best = (min if low else max)(cand, key=lambda m: v[m])
        L.append(f'| {i} | {lab} | ' + ' | '.join(f'**{f.format(v[m])}**' if m == best else f.format(v[m])
                                                   for m in methods) + ' |')
    return '\n'.join(L)


def per_job(res, plant, key, f, low=True, methods=MAIN):
    methods = [m for m in methods if m in res[plant][TEST_JOBS[0]]]
    rivals = [m for m in methods if m not in ('DP', 'PIRL-P')]
    L = ['| Công việc | ' + ' | '.join(methods) + ' | PIRL thắng |', '|' + '---|' * (len(methods) + 2)]
    wins = {m: 0 for m in rivals}
    for j in ['train_mixed'] + TEST_JOBS:
        R = res[plant][j]
        v = {m: agg(R, m, key) for m in methods}
        cand = [m for m in methods if m != 'DP']
        best = (min if low else max)(cand, key=lambda m: v[m][0])
        beat = [m for m in rivals if (v['PIRL-P'][0] < v[m][0] if low else v['PIRL-P'][0] > v[m][0])]
        if j != 'train_mixed':
            for m in beat:
                wins[m] += 1
        cells = []
        for m in methods:
            c = f.format(v[m][0]) + (f' ± {f.format(v[m][1])}' if len(R[m]) > 1 else '')
            cells.append(f'**{c}**' if m == best else c)
        name = j + (' *(validation)*' if j == 'train_mixed' else '')
        L.append(f'| {name} | ' + ' | '.join(cells) + f' | {len(beat)}/{len(rivals)} |')
    L.append('| **Số công việc PIRL thắng (/10)** | ' + ' | '.join(
        '–' if m in ('DP', 'PIRL-P') else f'**{wins[m]}**' for m in methods) + ' | |')
    return '\n'.join(L), wins


def training_table(R):
    groups = {}
    sel = json.load(open(os.path.join(RES_DIR, 'selected.json')))
    for name, tags in sel.items():
        groups[name] = [R[t] for t in tags]
    L = ['| Thuật toán | γ | Seed | Thời gian train (phút) | Vi phạm ràng buộc khi train | T pin max khi train (°C) '
         '| Episode hội tụ (≤ 2% chi phí cuối) |', '|---|---|---|---|---|---|---|']
    for name, ms in groups.items():
        conv = []
        for m in ms:
            c = np.array([e['val_cost'] for e in m['log']])
            conv.append(int(np.argmax(c <= 1.02 * c[-5:].mean())))
        L.append(f"| {name} | {ms[0]['gamma']} | {len(ms)} | {np.mean([m['train_time_s'] for m in ms]) / 60:.1f} "
                 f"| {np.mean([sum(e['train_interv'] for e in m['log']) for m in ms]):.0f} "
                 f"| {max(max(e['train_T_max'] for e in m['log']) for m in ms):.1f} | {np.mean(conv):.1f} |")
    return '\n'.join(L), groups


# ------------------------------------------------------------------ figures
def fig_gap(res, path):
    ms = [m for m in MAIN if m != 'DP' and m in res['nominal'][TEST_JOBS[0]]]
    fig, ax = plt.subplots(figsize=(11, 3.8))
    w = 0.84 / len(ms)
    x = np.arange(len(TEST_JOBS))
    for i, m in enumerate(ms):
        ax.bar(x + (i - (len(ms) - 1) / 2) * w, [agg(res['nominal'][j], m, 'gap_dp')[0] for j in TEST_JOBS],
               w * 0.9, color=COLORS[m], label=m)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(x, [j.replace('_', '\n') for j in TEST_JOBS])
    ax.set_ylabel('g/kWh công vượt DP (%)')
    ax.set_title('Nhiên liệu trên mỗi kWh công thủy lực, so với DP (thấp hơn = tốt hơn)', loc='left', color=INK)
    ax.legend(ncol=len(ms), loc='upper left')
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_learning(groups, path):
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.4))
    for name, ms in groups.items():
        if name not in ('PIRL', 'PIRL-P+', 'SAC', 'TD3', 'DDPG'):
            continue
        c = np.array([[e['val_cost'] for e in m['log']] for m in ms])
        iv = np.array([[e['train_interv'] for e in m['log']] for m in ms])
        ep = np.arange(1, c.shape[1] + 1)
        axs[0].plot(ep, c.mean(0), color=COLORS[name], lw=2, label=name)
        axs[0].fill_between(ep, c.min(0), c.max(0), color=COLORS[name], alpha=0.15, lw=0)
        axs[1].plot(ep, iv.mean(0), color=COLORS[name], lw=2, label=name)
    axs[0].set_title('Chi phí vận hành trên chu trình validation', loc='left', color=INK)
    axs[0].set_xlabel('Episode'); axs[0].set_ylabel('$/h'); axs[0].legend()
    axs[1].set_title('Số bước action bị supervisor sửa khi train', loc='left', color=INK)
    axs[1].set_xlabel('Episode'); axs[1].set_ylabel('bước / episode')
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_traces(path, job='trenching'):
    tr = np.load(os.path.join(RES_DIR, 'traces.npz'))
    ms = [m for m in ['DP', 'PIRL-P', 'TD3', 'A-ECMS'] if f'nominal|{job}|{m}' in tr]
    x0 = tr[f'nominal|{job}|{ms[0]}']
    t = np.arange(x0.shape[1]) * 0.5
    sl = slice(0, 240)
    fig, axs = plt.subplots(4, 1, figsize=(11, 8.4), sharex=False)
    for ax in axs[:2]:
        low = x0[9][sl] > 0.5
        ax.fill_between(t[sl], 0, 1, where=low, transform=ax.get_xaxis_transform(), color=GRID, lw=0)
    for m in ms:
        x = tr[f'nominal|{job}|{m}']
        axs[0].plot(t[sl], x[5][sl], color=COLORS[m], lw=1.6, label=m)
        axs[1].plot(t[sl], x[7][sl], color=COLORS[m], lw=1.4)
        axs[2].plot(t, x[0], color=COLORS[m], lw=1.6)
        axs[3].plot(t, x[1], color=COLORS[m], lw=1.6)
    axs[0].set_ylabel('Tốc độ ICE (rpm)', fontsize=8); axs[1].set_ylabel('Tốc độ EMG (rpm)', fontsize=8)
    axs[2].set_ylabel('SOC', fontsize=8); axs[3].set_ylabel('T pin (°C)', fontsize=8)
    axs[0].set_title(f'{job}: 120 s đầu (nền xám = pha hạ cần M2) và toàn chu trình (SOC, nhiệt độ)',
                     loc='left', color=INK)
    axs[0].legend(ncol=len(ms), loc='upper right'); axs[-1].set_xlabel('Thời gian (s)')
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_battery(res, path):
    ms = [m for m in MAIN if m in res['nominal'][TEST_JOBS[0]]]
    keys = [('aging_Ah_h', 'Hao mòn pin (Ah hiệu dụng/h)'), ('ohmic_kJ', 'Tổn hao I²R (kJ)'),
            ('T_max', 'T pin max (°C)'), ('rec_util_pct', 'Tận dụng hạ cần (%)')]
    fig, axs = plt.subplots(1, 4, figsize=(12, 3.1))
    for ax, (k, lab) in zip(axs, keys):
        v = [np.mean([agg(res['mismatch'][j], m, k)[0] for j in TEST_JOBS]) for m in ms]
        ax.bar(range(len(ms)), v, color=[COLORS[m] for m in ms], width=0.7)
        ax.set_xticks(range(len(ms)), ms, rotation=35, ha='right')
        ax.set_title(lab, loc='left', color=INK); ax.grid(axis='x', visible=False)
        if k == 'T_max':
            ax.set_ylim(35, max(v) + 2)
    fig.suptitle('Chỉ số pin, kịch bản mismatch (trời 45 °C, pin lão hoá), TB 10 công việc', x=0.01, ha='left', color=INK)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    R = runs()
    res = load()
    tt, groups = training_table(R)
    out = {
        'summary_nominal': summary(res, 'nominal', MAIN),
        'summary_mismatch': summary(res, 'mismatch', MAIN),
        'training': tt,
    }
    for plant in ('nominal', 'mismatch'):
        for key, f, low in [('sfc_work', '{:.1f}', True), ('unmet_pct', '{:.2f}', True),
                            ('fuel_eq_Lh', '{:.2f}', True), ('cost_usd_h', '{:.2f}', True),
                            ('aging_Ah_h', '{:.1f}', True)]:
            t, w = per_job(res, plant, key, f, low)
            out[f'per_job_{key}_{plant}'] = t
    with open(os.path.join(RES_DIR, 'tables.md'), 'w') as fo:
        for k, v in out.items():
            fo.write(f'### {k}\n\n{v}\n\n')
    fig_gap(res, os.path.join(RES_DIR, 'fig_gap_to_dp.png'))
    fig_learning(groups, os.path.join(RES_DIR, 'fig_learning.png'))
    fig_traces(os.path.join(RES_DIR, 'fig_traces_trenching.png'))
    fig_battery(res, os.path.join(RES_DIR, 'fig_battery.png'))
    print(open(os.path.join(RES_DIR, 'tables.md')).read())


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'select':
        select()
    else:
        main()
