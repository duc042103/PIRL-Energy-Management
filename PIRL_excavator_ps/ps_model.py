# -*- coding: utf-8 -*-
"""
May xuc hybrid POWER-SPLIT (so do: Carrier = ICE, Sun = truc ra -> bom, Ring = EMG).

   ICE --CE--[i_e]-- Carrier ┐
                             ├── bo hanh tinh (k = Z_R/Z_S = 78/30) ── Sun --[i_p]--C1-- Bom -- van chinh -- xy-lanh can
   EMG --CM--[i_m]-- Ring  ──┘                                                                     │ ha can
    │  └──C2── Motor TL (HM, luu luong co dinh) <── van thu hoi <──────────────────────────────────┘
    └── Pin Li-ion (DC bus)

Hai che do (chuyen bang luat, theo tin hieu ha can):
  M1 NANG / LAM VIEC : CM dong, C2 mo. Power-split: ICE + EMG dan bom qua bo hanh tinh.
  M2 HA CAN          : van chinh dong (bom khong cap dau), CM mo, C2 dong.
                       Dau hoi quay HM -> EMG phat dien -> nap pin. ICE chay khong tai.

M1: bo hanh tinh 3 phan tu -> ti so mo-men co dinh:
   T_Sun = T_bom / i_p,  T_Carrier = (1+k) T_Sun,  T_Ring = k T_Sun
   => mo-men ICE va EMG do tai bom quyet dinh; EMS chon TOC DO (n_bom, n_ICE):
   n_Sun = (1+k) n_Carrier - k n_Ring (Willis). n_ICE cao -> EMG phat dien; n_ICE thap -> EMG tro luc.
M2: EMS chi chon toc do khong tai cua ICE (thap = it nhien lieu, nhung phai kip tang toc cho lan nang sau).

Action = (n_bom, n_ICE). State s = [SOC, T_pin, n_ICE, n_bom, n_mode, P_hyd, P_boom, w_hm(khong dung)].
"""
import os
import sys
import math
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'PIRL_excavator'))
from excavator_model import (ExcavatorPhysics, LHV, RHO_DIESEL, rpm2rad, _interp1,   # noqa: E402,F401
                             ENG_N, ENG_TMAX)

S_DIM = 8


class PowerSplitPhysics(ExcavatorPhysics):
    """Ke thua pin (OCV, R, nhiet, lao hoa), ban do nhien lieu, hieu suat bom tu mo hinh truoc."""

    def __init__(self, **kw):
        super().__init__(**kw)
        # ---- bo hanh tinh + hop so
        self.k = 78.0 / 30.0              # Z_R / Z_S
        self.i_e = 1.6                    # ICE -> Carrier (giam toc)
        self.i_p = 2.4                    # Sun -> bom (giam toc)
        self.i_m = 3.0                    # EMG -> Ring (giam toc)
        self.eta_g, self.eta_pg = 0.98, 0.97
        # ---- dong co 110 kW, 900-2200 rpm; tai phu khi M2 (bom o luu luong cho + ma sat hop so)
        self.ne_min, self.ne_max = 900.0, 2200.0
        self.P_standby = 2000.0
        # ---- EMG PMSM 50 kW dinh / 260 Nm / 6000 rpm
        self.m1 = dict(T_max=260.0, P_max=50e3, kc=0.030, ki=1.2, kw=3e-5, C0=150.0)
        self.n_emg_max = 6000.0
        # ---- bom: 1000-2200 rpm, gioi han mo-men vao 620 Nm
        self.np_min, self.np_max, self.T_pump_max = 1000.0, 2200.0, 620.0
        # ---- motor thuy luc thu hoi: luu luong co dinh 63 cm^3/vong, noi thang EMG qua C2
        self.D_hm = 63e-6                 # m^3/vong
        self.p_boom = 10e6                # Pa, ap suat dau hoi ha can
        # ---- quan tinh (kg m^2)
        self.J_e, self.J_p, self.J_m = 1.2, 0.3, 0.08
        self.dn_max = 300.0

    # ------------------------------------------------------------------ dong hoc
    def kin(self, n_p, n_e):
        n_S = self.i_p * n_p
        n_R = ((1 + self.k) * n_e / self.i_e - n_S) / self.k
        return n_S, n_R, self.i_m * n_R

    def n_e_neutral(self, n_p):
        """Toc do ICE lam EMG dung yen o M1 (khong trao doi cong suat voi pin)."""
        return self.i_e * self.i_p * n_p / (1 + self.k)

    def emg_T_max(self, w):
        return torch.clamp(self.m1['P_max'] / torch.clamp(w.abs(), min=1.0), max=self.m1['T_max'])

    def emg_loss(self, T, w):
        m = self.m1
        wa = torch.sqrt(w ** 2 + 1.0)
        return m['kc'] * T ** 2 + m['ki'] * wa + m['kw'] * wa ** 3 + m['C0']

    @staticmethod
    def lowering(s):
        return s[..., 6] > 100.0

    # ------------------------------------------------------------------ M1: power-split
    def _m1(self, s, n_p, n_e):
        soc, Tb, ne0, np0, n_mode, P_hyd, P_boom, _ = s.unbind(-1)
        dt = self.dt
        wp, we = rpm2rad(n_p), rpm2rad(n_e)
        _, _, nm = self.kin(n_p, n_e)
        _, _, nm0 = self.kin(np0, ne0)
        wm, wm0 = rpm2rad(nm), rpm2rad(nm0)
        # bom: luu luong toi da ~ n_bom, gioi han mo-men vao
        P_del = torch.minimum(P_hyd, self.k_pump * n_p)
        unmet = torch.relu(P_hyd - P_del)
        P_pm = self.pump_mech(n_p, P_del) + self.J_p * (wp ** 2 - rpm2rad(np0) ** 2) / (2 * dt)
        T_P = P_pm / wp
        T_Pc = torch.clamp(T_P, max=self.T_pump_max)
        unmet = unmet + torch.relu(T_P - T_Pc) * wp * 0.85
        # bo hanh tinh (tinh, co hieu suat an khop)
        T_S = T_Pc / (self.i_p * self.eta_g)
        T_eng = (1 + self.k) * T_S / (self.eta_pg * self.i_e * self.eta_g) \
            + self.J_e * (we ** 2 - rpm2rad(ne0) ** 2) / (2 * dt) / we
        T_g = self.k * T_S / (self.i_m * self.eta_g)            # mo-men phan luc EMG phai giu
        T_emax = _interp1(ENG_N, ENG_TMAX, n_e)
        unmet = unmet + torch.relu(T_eng - T_emax) * we * 0.85
        T_eng = torch.minimum(torch.clamp(T_eng, min=0.0), T_emax)
        # EMG: cong suat co hoc hap thu (chieu phat dien duong)
        P_abs = T_g * wm - self.J_m * (wm ** 2 - wm0 ** 2) / (2 * dt)
        P_raw = self.P_aux + self.emg_loss(T_g, wm) - P_abs       # + = xa pin
        return dict(P_raw=P_raw, unmet=unmet, T_eng=T_eng, we=we, n_emg=nm, T_emg=T_g, P_pump=P_pm,
                    P_rec=torch.zeros_like(P_raw))

    # ------------------------------------------------------------------ M2: ha can, thu hoi
    def _m2(self, s, n_p, n_e):
        P_boom = s[..., 6]
        we = rpm2rad(n_e)
        Q_b = P_boom / self.p_boom                                   # m^3/s
        nm = torch.clamp(Q_b / self.D_hm * 60.0, max=self.n_emg_max)  # EMG quay theo HM
        wm = rpm2rad(nm)
        frac = torch.clamp(nm / torch.clamp(Q_b / self.D_hm * 60.0, min=1e-6), max=1.0)
        eta_hm = 0.90 - 0.15 * torch.exp(-wm / 60.0)
        P_hm = eta_hm * P_boom * frac
        T_g = torch.minimum(P_hm / torch.clamp(wm, min=1.0), self.emg_T_max(wm))
        P_hm = T_g * wm
        P_raw = self.P_aux + self.emg_loss(T_g, wm) - P_hm
        T_eng = self.P_standby / we
        return dict(P_raw=P_raw, unmet=torch.zeros_like(P_raw), T_eng=T_eng, we=we, n_emg=nm, T_emg=T_g,
                    P_pump=torch.full_like(P_raw, self.P_standby), P_rec=P_hm, eta_hm=eta_hm)

    def _core(self, s, n_p, n_e):
        a, b = self._m1(s, n_p, n_e), self._m2(s, n_p, n_e)
        low = self.lowering(s)
        return {k: torch.where(low, b[k], a[k]) for k in a}, low

    # ------------------------------------------------------------------ mien kha thi
    def pump_window(self, s, pump_req=False):
        np0, n_mode, P_hyd = s[..., 3], s[..., 4], s[..., 5]
        cap = torch.clamp(n_mode, self.np_min, self.np_max)
        lo = torch.clamp(np0 - self.dn_max, min=self.np_min)
        hi = torch.maximum(torch.minimum(cap, np0 + self.dn_max), lo)
        if pump_req:
            lo = torch.minimum(torch.maximum(lo, P_hyd / self.k_pump), hi)
        return lo, hi

    def engine_bounds(self, s, n_p, iters=8):
        """[n_e_lo, n_e_hi]: toc do doi +-300 rpm; o M1 them gioi han mo-men ICE, toc do EMG,
        gioi han xa/nap pin (P_pin giam theo n_ICE -> giai bang chia doi)."""
        with torch.no_grad():
            ne0 = s[..., 2]
            r_lo = torch.clamp(ne0 - self.dn_max, min=self.ne_min)
            r_hi = torch.maximum(torch.clamp(ne0 + self.dn_max, max=self.ne_max), r_lo)
            lo, hi = r_lo.clone(), r_hi.clone()
            # toc do EMG |n_emg| <= 6000
            n_S = self.i_p * n_p
            nR = self.n_emg_max / self.i_m
            lo = torch.maximum(lo, (n_S - self.k * nR) * self.i_e / (1 + self.k))
            hi = torch.minimum(hi, (n_S + self.k * nR) * self.i_e / (1 + self.k))
            # mo-men ICE yeu cau <= T_max(n): nhanh tang 800-1400 rpm, nhanh giam 1500-2200 rpm
            P_del = torch.minimum(s[..., 5], self.k_pump * n_p)
            T_P = torch.clamp(self.pump_mech(n_p, P_del) / rpm2rad(n_p), max=self.T_pump_max)
            T_need = (1 + self.k) * T_P / (self.i_p * self.eta_g * self.eta_pg * self.i_e * self.eta_g)
            up_T, up_n = ENG_TMAX[:7], ENG_N[:7]
            dn_T, dn_n = ENG_TMAX[7:].flip(0), ENG_N[7:].flip(0)
            lo = torch.maximum(lo, torch.where(T_need <= up_T[0], torch.full_like(T_need, 800.0),
                                               _interp1(up_T, up_n, T_need)))
            hi = torch.minimum(hi, torch.where(T_need <= dn_T[0], torch.full_like(T_need, 2200.0),
                                               _interp1(dn_T, dn_n, T_need)))
            # pin: o M1, P_pin(n_ICE) chi phu thuoc toc do EMG (tuyen tinh theo n_ICE) va T_EMG (khong doi)
            P_b_min, P_b_max = self.batt_limits(s[..., 0], s[..., 1])
            T_g = self.k * T_P / (self.i_p * self.eta_g) / (self.i_m * self.eta_g)
            wm0 = rpm2rad(self.kin(s[..., 3], s[..., 2])[2])
            J_m, dt = self.J_m, self.dt

            def g(n):
                wm = rpm2rad(self.kin(n_p, n)[2])
                return self.P_aux + self.emg_loss(T_g, wm) - T_g * wm + J_m * (wm ** 2 - wm0 ** 2) / (2 * dt)

            def bisect(ok_at, a, b, want_low):
                x_lo, x_hi = a.clone(), b.clone()
                for _ in range(iters):
                    m = 0.5 * (x_lo + x_hi)
                    ok = ok_at(m)
                    if want_low:          # n nho nhat thoa
                        x_hi, x_lo = torch.where(ok, m, x_hi), torch.where(ok, x_lo, m)
                    else:                 # n lon nhat thoa
                        x_lo, x_hi = torch.where(ok, m, x_lo), torch.where(ok, x_hi, m)
                return x_hi if want_low else x_lo

            dis_ok = lambda n: g(n) <= P_b_max                  # noqa: E731  (n cao -> xa it)
            chg_ok = lambda n: g(n) >= P_b_min                  # noqa: E731  (n thap -> nap it)
            lo_b = torch.where(dis_ok(r_lo), r_lo, torch.where(~dis_ok(r_hi), r_hi,
                                                                bisect(dis_ok, r_lo, r_hi, True)))
            hi_b = torch.where(chg_ok(r_hi), r_hi, torch.where(~chg_ok(r_lo), r_lo,
                                                                bisect(chg_ok, r_lo, r_hi, False)))
            lo = torch.maximum(lo, lo_b)
            hi = torch.minimum(hi, hi_b)
            lo = torch.minimum(torch.maximum(lo, r_lo), r_hi)
            hi = torch.maximum(torch.minimum(hi, r_hi), lo)
            low = self.lowering(s)
            lo, hi = torch.where(low, r_lo, lo), torch.where(low, r_hi, hi)     # M2: chi gioi han toc do doi
        return lo, hi

    # ------------------------------------------------------------------ 1 buoc mo phong
    def step_detail(self, s, n_p, n_e):
        soc, Tb = s[..., 0], s[..., 1]
        c, low = self._core(s, n_p, n_e)
        V, R = self.ocv(soc), self.resistance(soc, Tb)
        P_b_min, P_b_max = self.batt_limits(soc, Tb)
        P_raw = c['P_raw']
        curtail = torch.minimum(torch.relu(P_b_min - P_raw), c['P_rec'])     # van thu hoi xa bot
        dump = torch.relu(P_b_min - P_raw - curtail)                          # M1: ICE qua nhanh
        P_batt = P_raw + curtail + dump
        deficit = torch.relu(P_batt - P_b_max)                                # pin khong du -> thieu cong
        P_batt = P_batt - deficit
        I = (V - torch.sqrt(torch.clamp(V ** 2 - 4 * R * P_batt, min=1.0))) / (2 * R)
        dt = self.dt
        soc_n = soc - I * dt / (3600.0 * self.Q_Ah)
        Tb_n = Tb + dt / self.C_th * (I ** 2 * R - self.hA * (Tb - self.T_amb))
        P_eng = c['T_eng'] * c['we']
        fuel = self.fuel_rate(n_e, P_eng) * dt
        I_abs = torch.sqrt(I ** 2 + 1e-2)
        sev = torch.exp(self.Ea / 8.314 * (1 / 298.15 - 1 / (Tb + 273.15))) * \
            (1 + 0.3 * torch.relu(I_abs / self.Q_Ah - 1))
        age_Ah = I_abs * dt / 3600.0 * sev
        rec_avail = torch.where(low, 0.88 * s[..., 6], torch.zeros_like(P_raw))   # tham chieu: HM 88%
        rec_used = c['P_rec'] - curtail
        return dict(soc=soc_n, Tb=Tb_n, fuel=fuel, age_Ah=age_Ah, I=I, V=V, R=R, P_batt=P_batt,
                    P_pump=c['P_pump'], P_eng=P_eng, P_rec=rec_avail, P_rec_used=rec_used,
                    curtail=torch.relu(rec_avail - rec_used), unmet=(c['unmet'] + deficit) * dt,
                    dump=dump * dt, n_emg=c['n_emg'], T_eng=c['T_eng'], T_emg=c['T_emg'],
                    mode=low.to(P_raw.dtype))


PHYS = PowerSplitPhysics()


def plant_mismatch():
    return PowerSplitPhysics(eta_eng=0.97, pump_eff_offset=-0.03, R_scale=1.4, Q_scale=0.85, T_amb=45.0)


BETA_SOC = 1000.0


def reward(d, phys=PHYS):
    return -(d['fuel'] + phys.W_AGE * d['age_Ah'] + BETA_SOC * (d['soc'] - phys.soc_ref) ** 2
             + phys.W_UNMET * d['unmet'])


def project(s, n_p, n_e, phys=PHYS):
    """Supervisor: chieu (n_bom, n_ICE) vao mien kha thi cua may."""
    lo, hi = phys.pump_window(s)
    n_p = torch.maximum(torch.minimum(n_p, hi), lo)
    elo, ehi = phys.engine_bounds(s, n_p)
    return n_p, torch.maximum(torch.minimum(n_e, ehi), elo)
