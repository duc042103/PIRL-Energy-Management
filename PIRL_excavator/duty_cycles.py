# -*- coding: utf-8 -*-
"""
Chu trinh tai (duty cycle) cua may xuc 20 tan, buoc thoi gian 0.5 s.

Moi chu trinh la chuoi cac pha thao tac; moi pha co:
  n      toc do dong co theo che do lam viec (rpm)
  P_hyd  cong suat thuy luc bom phai cap (W)
  P_boom cong suat thuy luc cua dau hoi ve khi HA CAN (W) -> co the thu hoi
  w_hm   toc do motor thuy luc thu hoi (rad/s) ~ luu luong dau ha can

Train: 'train_mixed' (tron ngau nhien dao-do tai + san gat + dao ranh + cho, moi episode 1 seed moi).
Test : 10 loai cong viec co dinh (seed co dinh), gom ca cong viec KHONG co trong luc train.
"""
import zlib
import numpy as np

DT = 0.5
TEST_JOBS = ['heavy_dig', 'truck_load_180', 'trenching', 'grading', 'pipe_lifting',
             'breaker', 'dig_with_waiting', 'eco_dig', 'dig_and_travel', 'rock_excavation']


class _Builder:
    def __init__(self, rng):
        self.rng = rng
        self.segs = []          # (duration_s, n, P_hyd_kW, P_boom_kW, w_hm)

    def add(self, dur, n, P, Pb=0.0, w=0.0, jitter=0.12):
        r = self.rng
        dur = max(DT, dur * (1 + r.uniform(-0.2, 0.2)))
        P = max(0.0, P * (1 + r.uniform(-jitter, jitter)))
        Pb = max(0.0, Pb * (1 + r.uniform(-jitter, jitter)))
        self.segs.append((dur, n, P, Pb, w))

    def dig_cycle(self, n, dig=85, lift=65, dump=25, back=30, boom_rec=28, swing=4.0, dig_t=4.5,
                  down_t=4.0, w=150.0):
        self.add(dig_t, n, dig, jitter=0.18)                     # dao (xuc dat)
        self.add(swing, n, lift)                                 # nang can + quay
        self.add(1.5, n, dump)                                   # do tai
        self.add(down_t, n, back, boom_rec, w)                   # quay ve + HA CAN (thu hoi)
        self.add(0.7, n, 8)                                      # nghi ngan

    def idle(self, dur):
        self.add(dur, 1000, 3, jitter=0.3)                       # auto-idle 1000 rpm

    def build(self, total_s):
        t, out = 0.0, []
        for dur, n, P, Pb, w in self.segs:
            k = int(round(dur / DT))
            out += [(n, P, Pb, w)] * k
            t += k * DT
            if t >= total_s:
                break
        a = np.array(out[:int(total_s / DT)], dtype=np.float64)
        # loc quan tinh bac 1 (tau = 0.6 s) + nhieu dao dong ap suat
        y = a.copy()
        alpha = DT / (0.6 + DT)
        for i in range(1, len(a)):
            y[i, 1:3] = y[i - 1, 1:3] + alpha * (a[i, 1:3] - y[i - 1, 1:3])
        y[:, 1] *= 1 + 0.05 * self.rng.standard_normal(len(y))
        y[:, 1:3] = np.clip(y[:, 1:3], 0, None)
        return {'n': y[:, 0], 'P_hyd': y[:, 1] * 1e3, 'P_boom': y[:, 2] * 1e3,
                'w_hm': np.where(y[:, 2] > 0.5, y[:, 3], 0.0)}


def make_cycle(job, seed=0, total_s=600.0):
    rng = np.random.default_rng(zlib.crc32(job.encode()) % 10_000 + 7919 * seed)
    b = _Builder(rng)
    while sum(s[0] for s in b.segs) < total_s + 30:
        if job == 'heavy_dig':                 # dat cung, che do H 2000 rpm
            b.dig_cycle(2000, dig=105, lift=80, back=32, boom_rec=32)
        elif job == 'truck_load_180':          # do len xe tai, quay 180 do
            b.dig_cycle(1800, dig=85, lift=70, swing=6.5, back=40, down_t=5.5, boom_rec=26)
        elif job == 'trenching':               # dao ranh sau, ha can dai
            b.dig_cycle(1800, dig=75, lift=60, back=28, down_t=6.5, boom_rec=35, w=170)
        elif job == 'grading':                 # san gat: tai lien tuc thap, ha can nhe thuong xuyen
            b.add(6, 1600, 42); b.add(3, 1600, 30, 9, 80); b.add(5, 1600, 48); b.add(2, 1600, 25, 6, 60)
        elif job == 'pipe_lifting':            # cau ong: nang cham tai nang, ha cham co tai (thu hoi lon)
            b.add(9, 1600, 55); b.add(10, 1600, 12); b.add(9, 1600, 18, 38, 120); b.add(6, 1600, 10)
        elif job == 'breaker':                 # bua pha da: tai cao on dinh, KHONG thu hoi
            b.add(20, 1800, 68, jitter=0.1); b.add(3, 1800, 30, 6, 60)
        elif job == 'dig_with_waiting':        # cho xe tai: xen ke khong tai dai
            for _ in range(4):
                b.dig_cycle(1800)
            b.idle(rng.uniform(30, 60))
        elif job == 'eco_dig':                 # che do E 1600 rpm
            b.dig_cycle(1600, dig=70, lift=55, back=26, boom_rec=24)
        elif job == 'dig_and_travel':          # di chuyen xen dao
            for _ in range(3):
                b.dig_cycle(1800)
            b.add(rng.uniform(20, 35), 1900, 92, jitter=0.1)
        elif job == 'rock_excavation':         # dinh tai cao, bien dong lon
            b.add(3, 2000, 125, jitter=0.1); b.add(2, 2000, 70); b.dig_cycle(2000, dig=110, lift=85, boom_rec=34)
        elif job == 'train_mixed':             # du lieu train
            r = rng.uniform()
            if r < 0.45:
                b.dig_cycle(rng.choice([1700, 1800, 1900]), dig=rng.uniform(70, 95),
                            lift=rng.uniform(55, 75), back=rng.uniform(25, 38),
                            boom_rec=rng.uniform(20, 32), swing=rng.uniform(3.5, 6))
            elif r < 0.65:
                b.add(6, 1700, 40); b.add(3, 1700, 28, 8, 80)
            elif r < 0.85:
                b.dig_cycle(1800, dig=78, lift=60, down_t=6, boom_rec=32, w=165)
            else:
                b.idle(rng.uniform(10, 40))
        else:
            raise ValueError(job)
    return b.build(total_s)


if __name__ == '__main__':
    for j in ['train_mixed'] + TEST_JOBS:
        c = make_cycle(j)
        E_hyd = c['P_hyd'].sum() * DT / 3.6e6
        E_boom = c['P_boom'].sum() * DT / 3.6e6
        print(f"{j:18s} N={len(c['n'])} P_hyd mean {c['P_hyd'].mean()/1e3:5.1f} kW max "
              f"{c['P_hyd'].max()/1e3:5.1f} | E_hyd {E_hyd:5.2f} kWh E_boom {E_boom:4.2f} kWh "
              f"({100*E_boom/E_hyd:4.1f}%)")
