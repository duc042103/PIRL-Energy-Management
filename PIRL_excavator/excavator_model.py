# -*- coding: utf-8 -*-
"""
Mo hinh vat ly may xuc hybrid 20 tan (song song) + he thu hoi nang luong can (boom).

                 +---------- truc bom ----------+
   Dong co diesel ==== EMG1 (motor/may phat) ==== Bom thuy luc chinh --> boom/arm/bucket/swing/travel
                         |                                                   |
                         |  DC bus                    ha can: dau hoi ve     v
   Pin Li-ion <==========+==========<  EMG2 (may phat)  <== motor thuy luc (HM)
   (nhiet + lao hoa)

Action cua EMS (2 chieu):
   (1) toc do dong co n (<= toc do che do do tai xe chon; gioi han toc do thay doi +-300 rpm / 0.5 s,
       co quan tinh truc; bom bien thien lu luong bu lai nhung luu luong toi da ti le voi n)
   (2) cong suat dong co P_eng. Khi do
   P_EMG1_mech = P_pump_mech - P_eng              (can bang cong suat tren truc)
   P_batt      = P_EMG1_elec + P_aux - P_rec      (can bang cong suat tren DC bus)
Nang luong thu hoi P_rec duoc uu tien nap vao pin. Neu pin khong nhan them duoc
(gioi han dong sac / SOC / nhiet) thi phan du bi tiet luu thanh nhiet (curtail).

Tat ca cong thuc viet bang torch -> kha vi -> nhung duoc vao mang NN cua PIRL.
"""
import math
import torch

LHV = 42800.0          # J/g diesel
RHO_DIESEL = 832.0     # g/L
RG = 8.314


def rpm2rad(n):
    return n * (2 * math.pi / 60)


def _interp1(xg, yg, x):
    """Noi suy tuyen tinh 1D (kha vi theo x)."""
    xg, yg = xg.to(x.dtype), yg.to(x.dtype)
    x = torch.clamp(x, float(xg[0]), float(xg[-1]))
    i = torch.clamp(torch.searchsorted(xg, x.contiguous()), 1, len(xg) - 1)
    x0, x1, y0, y1 = xg[i - 1], xg[i], yg[i - 1], yg[i]
    return y0 + (x - x0) / (x1 - x0) * (y1 - y0)


def _interp2(x0, dx, y0, dy, table, x, y):
    """Noi suy song tuyen tinh tren luoi deu (kha vi theo x, y)."""
    table = table.to(x.dtype)
    nx, ny = table.shape
    fx = torch.clamp((x - x0) / dx, 0, nx - 1 - 1e-6)
    fy = torch.clamp((y - y0) / dy, 0, ny - 1 - 1e-6)
    ix, iy = fx.floor().long(), fy.floor().long()
    wx, wy = fx - ix, fy - iy
    t00, t10 = table[ix, iy], table[ix + 1, iy]
    t01, t11 = table[ix, iy + 1], table[ix + 1, iy + 1]
    return (t00 * (1 - wx) * (1 - wy) + t10 * wx * (1 - wy) + t01 * (1 - wx) * wy + t11 * wx * wy)


# =============================================================================
# BAN DO (MAP) - sinh tu mo hinh ban vat ly, luu duoi dang bang nhu du lieu bench-test
# =============================================================================
ENG_N = torch.arange(800.0, 2201.0, 100.0)                      # rpm
ENG_T = torch.arange(0.0, 701.0, 25.0)                           # Nm
ENG_TMAX = torch.tensor([400, 470, 520, 570, 610, 635, 650, 650, 645, 640,
                         625, 600, 565, 525, 480], dtype=torch.float32)   # Nm theo ENG_N


def _engine_fuel_table(eta_scale=1.0):
    """Ban do suat tieu hao nhien lieu (g/s) dong co diesel 5.2 L, 110 kW @ 2000 rpm.
    Willans: m_dot = (P_eff + P_fric(n)) / (eta_ind(n, tau) * LHV)."""
    n, T = ENG_N[:, None], ENG_T[None, :]
    w = rpm2rad(n)
    tau = T / ENG_TMAX[:, None]
    P_fric = (30.0 + 0.02 * (n - 800.0)) * w
    eta = eta_scale * (0.44 - 0.05 * ((n - 1400.0) / 800.0) ** 2 - 0.04 * (tau - 0.7) ** 2)
    return (T * w + P_fric) / (eta * LHV)


class ExcavatorPhysics:
    """Tat ca phuong trinh vat ly (khong co tham so hoc)."""

    def __init__(self, eta_eng=1.0, pump_eff_offset=0.0, R_scale=1.0, Q_scale=1.0,
                 T_amb=35.0):
        self.dt = 0.5
        # ---- dong co
        self.fuel_table = _engine_fuel_table(eta_eng)
        self.P_eng_rated = 110e3
        # ---- EMG1 (PMSM 45 kW, gan truc bom): ton hao = kc*T^2 + ki*w + kw*w^3 + C0
        self.m1 = dict(T_max=350.0, P_max=44e3, kc=0.030, ki=1.2, kw=3e-5, C0=150.0)
        # ---- EMG2 (may phat thu hoi 30 kW) + motor thuy luc
        self.m2 = dict(P_max=30e3, kc=0.045, ki=0.8, kw=2e-5, C0=100.0)
        # ---- bom + phu tai
        self.pump_eff_offset = pump_eff_offset
        self.P_aux = 1000.0
        # ---- toc do truc: quan tinh (dong co + EMG1 + bom), gioi han toc do thay doi, cong suat bom toi da
        self.J_shaft = 2.5            # kg m^2
        self.dn_max = 300.0           # rpm moi buoc 0.5 s
        self.n_min = 1000.0
        self.k_pump = 70.0            # W thuy luc toi da / rpm (luu luong toi da ~ n)
        self.W_UNMET = 0.19e-3        # g tuong duong / J thuy luc khong dap ung (mat nang suat)
        # ---- pin Li-ion NMC 96s, 25 Ah (~9 kWh)
        self.Q_Ah = 25.0 * Q_scale
        self.R_scale = R_scale
        self.I_dis, self.I_chg = 125.0, 75.0       # 5C / 3C
        self.soc_min, self.soc_max, self.soc_ref = 0.30, 0.80, 0.55
        self.ocv_soc = torch.linspace(0, 1, 11)
        self.ocv_cell = torch.tensor([3.00, 3.45, 3.55, 3.60, 3.65, 3.70, 3.78, 3.87, 3.96, 4.06, 4.18])
        # ---- nhiet pin: C dT/dt = I^2 R - hA (T - T_amb); giam dinh muc dong tu 40 -> 50 C
        self.C_th, self.hA, self.T_amb = 25e3, 12.0, T_amb
        self.T_derate, self.T_cut = 40.0, 50.0
        # ---- lao hoa: Ah-throughput co trong so nhiet do (Arrhenius) va C-rate
        self.Ea = 31500.0
        self.W_AGE = 7.85      # g diesel tuong duong cho 1 Ah "hieu dung" (gia pin / tuoi tho)

    # ------------------------------------------------------------------ thanh phan
    def pump_mech(self, n, P_hyd):
        eta = 0.92 - 0.12 * torch.exp(-P_hyd / 15e3) - 0.03 * ((n - 1600.0) / 600.0) ** 2 \
            + self.pump_eff_offset
        return P_hyd / eta

    def eng_P_max(self, n):
        return _interp1(ENG_N, ENG_TMAX, n) * rpm2rad(n)

    def fuel_rate(self, n, P_eng):
        T = P_eng / rpm2rad(n)
        return _interp2(800.0, 100.0, 0.0, 25.0, self.fuel_table, n, T)      # g/s

    def emg1_P_max(self, w):
        return torch.clamp(self.m1['T_max'] * w, max=self.m1['P_max'])

    def emg1_L0(self, w):
        m = self.m1
        return m['ki'] * w + m['kw'] * w ** 3 + m['C0']

    def emg1_elec(self, P_m, w):
        """Cong suat dien EMG1 (+ = lay tu bus, - = phat vao bus)."""
        return P_m + self.m1['kc'] * (P_m / w) ** 2 + self.emg1_L0(w)

    def recovery(self, P_boom, w_hm):
        """Cong suat dien thu hoi tu dau ha can: HM -> EMG2 -> bus."""
        eta_hm = 0.90 - 0.15 * torch.exp(-w_hm / 60.0)
        P_hm = eta_hm * P_boom
        w = torch.clamp(w_hm, min=5.0)
        m = self.m2
        loss = m['kc'] * (P_hm / w) ** 2 + m['ki'] * w + m['kw'] * w ** 3 + m['C0']
        P_rec = torch.clamp(P_hm - loss * (P_boom > 100).to(P_hm.dtype), min=0.0)
        return torch.clamp(P_rec, max=m['P_max'])

    def ocv(self, soc):
        return 96.0 * _interp1(self.ocv_soc, self.ocv_cell, soc)

    def resistance(self, soc, Tb):
        return self.R_scale * 0.12 * (1 + 0.8 * torch.relu(0.35 - soc) / 0.35) * \
            torch.exp(2500.0 * (1.0 / (Tb + 273.15) - 1.0 / 298.15))

    def batt_limits(self, soc, Tb):
        """Gioi han cong suat pin tu: dong 5C/3C, giam dinh muc theo nhiet, cua so SOC."""
        V, R = self.ocv(soc), self.resistance(soc, Tb)
        der = torch.clamp((self.T_cut - Tb) / (self.T_cut - self.T_derate), 0.0, 1.0)
        cap = self.Q_Ah * 3600.0 / self.dt
        I_max = torch.minimum(self.I_dis * der, torch.clamp((soc - self.soc_min) * cap, min=0.0))
        I_min = -torch.minimum(self.I_chg * der, torch.clamp((self.soc_max - soc) * cap, min=0.0))
        return V * I_min - R * I_min ** 2, V * I_max - R * I_max ** 2

    # ------------------------------------------------------------------ toc do
    def speed_window(self, s, pump_req=False):
        """s = [SOC, T_batt, n_cur, n_mode, P_hyd, P_boom, w_hm] -> [n_lo, n_hi] dat duoc trong 1 buoc.
        pump_req=True: nang n_lo de bom du luu luong cho P_hyd hien tai (neu dat duoc)."""
        n_cur, n_mode, P_hyd = s[..., 2], s[..., 3], s[..., 4]
        lo = torch.clamp(n_cur - self.dn_max, min=self.n_min)
        hi = torch.maximum(torch.minimum(n_mode, n_cur + self.dn_max), lo)
        if pump_req:
            lo = torch.minimum(torch.maximum(lo, P_hyd / self.k_pump), hi)
        return lo, hi

    def shaft_demand(self, s, n_next):
        """Cong suat co hoc can tren truc = bom (voi luu luong bi gioi han boi n) + quan tinh."""
        P_hyd = s[..., 4]
        P_del = torch.minimum(P_hyd, self.k_pump * n_next)
        P_in = self.J_shaft * (rpm2rad(n_next) ** 2 - rpm2rad(s[..., 2]) ** 2) / (2 * self.dt)
        return self.pump_mech(n_next, P_del) + P_in, torch.relu(P_hyd - P_del)

    # ------------------------------------------------------------------ mien kha thi
    def bounds(self, s, n):
        """Mien kha thi cua P_eng khi truc quay o toc do n (buoc ke tiep).
        Giai dong (closed-form) bat phuong trinh bac 2 cua ton hao EMG1."""
        soc, Tb, _, _, P_hyd, P_boom, w_hm = s.unbind(-1)
        w = rpm2rad(n)
        P_pump, _ = self.shaft_demand(s, n)
        P_rec = self.recovery(P_boom, w_hm)
        P_b_min, P_b_max = self.batt_limits(soc, Tb)
        a = self.m1['kc'] / w ** 2
        L0 = self.emg1_L0(w)

        def root(c):     # nghiem lon cua a x^2 + x + c = 0
            return (-1.0 + torch.sqrt(torch.clamp(1.0 - 4 * a * c, min=0.0))) / (2 * a)
        Pm_up = root(L0 - (P_b_max - self.P_aux + P_rec))     # gioi han xa (dung het P_rec)
        Pm_lo = root(L0 - (P_b_min - self.P_aux))             # gioi han sac (cho phep tiet luu P_rec)
        Pemg = self.emg1_P_max(w)
        P_eng_max = self.eng_P_max(n)
        lo = torch.maximum(torch.maximum(P_pump - Pemg, P_pump - Pm_up), torch.zeros_like(P_pump))
        hi = torch.minimum(torch.minimum(P_eng_max, P_pump + Pemg), P_pump - Pm_lo)
        lo = torch.minimum(lo, P_eng_max)
        hi = torch.maximum(hi, lo)
        return dict(lo=lo, hi=hi, P_pump=P_pump, P_rec=P_rec, P_b_min=P_b_min, P_b_max=P_b_max,
                    P_eng_max=P_eng_max)

    # ------------------------------------------------------------------ 1 buoc mo phong
    def step_detail(self, s, n, P_eng):
        soc, Tb, _, _, P_hyd, P_boom, w_hm = s.unbind(-1)
        w = rpm2rad(n)
        P_pump, unmet_hyd = self.shaft_demand(s, n)
        P_rec = self.recovery(P_boom, w_hm)
        P_m = P_pump - P_eng
        P_e1 = self.emg1_elec(P_m, w)
        V, R = self.ocv(soc), self.resistance(soc, Tb)
        P_b_min, P_b_max = self.batt_limits(soc, Tb)
        P_raw = P_e1 + self.P_aux - P_rec
        curtail = torch.clamp(torch.relu(P_b_min - P_raw), max=P_rec)   # thu hoi bi tiet luu
        P_batt = P_raw + curtail
        unmet = torch.relu(P_batt - P_b_max)
        P_batt = P_batt - unmet
        I = (V - torch.sqrt(torch.clamp(V ** 2 - 4 * R * P_batt, min=1.0))) / (2 * R)
        dt = self.dt
        soc_n = soc - I * dt / (3600.0 * self.Q_Ah)
        Tb_n = Tb + dt / self.C_th * (I ** 2 * R - self.hA * (Tb - self.T_amb))
        fuel = self.fuel_rate(n, P_eng) * dt
        I_abs = torch.sqrt(I ** 2 + 1e-2)
        sev = torch.exp(self.Ea / RG * (1 / 298.15 - 1 / (Tb + 273.15))) * \
            (1 + 0.3 * torch.relu(I_abs / self.Q_Ah - 1))
        age_Ah = I_abs * dt / 3600.0 * sev
        unmet_hyd = unmet_hyd + unmet                     # dien khong du -> bom cung thieu
        return dict(soc=soc_n, Tb=Tb_n, fuel=fuel, age_Ah=age_Ah, I=I, V=V, R=R, P_batt=P_batt,
                    P_pump=P_pump, P_rec=P_rec, curtail=curtail, unmet=unmet_hyd * dt, P_m=P_m,
                    P_e1=P_e1, n=n)


PHYS = ExcavatorPhysics()                     # mo hinh danh nghia (nhung trong PIRL)


def plant_mismatch():
    """May thuc khac mo hinh: dong co mon (eta -3%), bom mon (-3 diem %), pin lao hoa
    (R +40%, Q -15%), troi nong 45 C."""
    return ExcavatorPhysics(eta_eng=0.97, pump_eff_offset=-0.03, R_scale=1.4, Q_scale=0.85,
                            T_amb=45.0)


BETA_SOC = 1000.0


def reward(d, phys=PHYS):
    """r = -(nhien lieu + chi phi lao hoa pin + phat lech SOC + mat nang suat), don vi: g diesel tuong duong."""
    return -(d['fuel'] + phys.W_AGE * d['age_Ah'] + BETA_SOC * (d['soc'] - phys.soc_ref) ** 2
             + phys.W_UNMET * d['unmet'])
