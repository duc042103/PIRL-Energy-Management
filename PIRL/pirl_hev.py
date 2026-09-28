# -*- coding: utf-8 -*-
"""
PIRL - Physics-Informed Reinforcement Learning (vi du don gian)
Bai toan: quan ly nang luong xe hybrid (HEV) - chon cong suat dong co P_eng moi giay.

Vat ly duoc NHUNG vao mang NN o 3 cho:
  (1) PhysicsFeatureLayer : actor/critic nhan them dac trung vat ly (P_req, gioi han pin)
  (2) PhysicsConstraintLayer (lop cuoi cua Actor): anh xa output NN vao mien kha thi
      [P_eng_min, P_eng_max] tinh tu dong luc hoc xe + mo hinh pin  -> action luon hop le
  (3) Critic Q(s,a) = r_phys(s,a) + gamma * V_nn( f_phys(s,a) )
      - r_phys, f_phys la phuong trinh vat ly kha vi (khong hoc)
      - chi V_nn la phan hoc duoc -> gradient cua actor chay XUYEN qua vat ly

Chay:  python PIRL/pirl_hev.py
"""
import os
import copy
import numpy as np
import scipy.io as scio
import torch
import torch.nn as nn

torch.manual_seed(0)
np.random.seed(0)

HERE = os.path.dirname(os.path.abspath(__file__))
CYCLE_DIR = os.path.join(HERE, '..', 'Data_Standard Driving Cycles')


# =============================================================================
# 1. PHYSICS LAYER - mo hinh xe HEV don gian, viet bang torch nen kha vi
# =============================================================================
class HEVPhysics(nn.Module):
    """Tat ca cong thuc vat ly. Khong co tham so hoc - chi la phep tinh kha vi."""

    def __init__(self):
        super().__init__()
        # Xe (tham so Prius, giong Prius_model_new.py)
        self.m, self.g, self.Cr = 1449.0, 9.81, 0.013
        self.rho, self.A, self.Cd = 1.2, 2.23, 0.26
        self.eta_drive, self.eta_regen = 0.90, 0.60
        # Dong co: suat tieu hao nhien lieu dang bac 2 (g/s)
        self.P_eng_max = 56e3
        self.f0, self.f1, self.f2 = 0.25, 1 / (0.38 * 42600), 4e-10
        # Pin Ni-MH: mo hinh R_int
        self.Q = 6.5 * 3600          # dung luong (C)
        self.R = 0.37                # dien tro trong (ohm)
        self.I_dis, self.I_chg = 196.0, 120.0
        self.dt = 1.0
        self.soc_min, self.soc_max = 0.40, 0.80   # cua so SOC an toan cho pin

    def voc(self, soc):
        return 202.0 + 35.0 * soc    # OCV xap xi tuyen tinh theo SOC

    def power_demand(self, v, acc):
        """Can bang luc doc truc: F = m*a + lan + gio  ->  P_req tai bus (W)."""
        F = self.m * acc + self.m * self.g * self.Cr * (v > 0).float() \
            + 0.5 * self.rho * self.A * self.Cd * v ** 2
        P_wheel = F * v
        return torch.where(P_wheel >= 0, P_wheel / self.eta_drive, P_wheel * self.eta_regen)

    def battery_limits(self, soc):
        """Gioi han cong suat pin tu gioi han dong dien (P = Voc*I - R*I^2)
        va tu cua so SOC: SOC_next = SOC - I*dt/Q phai nam trong [soc_min, soc_max]."""
        V = self.voc(soc)
        I_max = torch.clamp((soc - self.soc_min) * self.Q / self.dt, 0.0, self.I_dis)
        I_min = -torch.clamp((self.soc_max - soc) * self.Q / self.dt, 0.0, self.I_chg)
        P_b_max = V * I_max - self.R * I_max ** 2
        P_b_min = V * I_min - self.R * I_min ** 2
        return P_b_min, P_b_max

    def engine_bounds(self, soc, v, acc):
        """Mien kha thi cua action: P_req = P_eng + P_batt, P_batt trong [P_b_min, P_b_max]."""
        P_req = self.power_demand(v, acc)
        P_b_min, P_b_max = self.battery_limits(soc)
        lo = torch.clamp(P_req - P_b_max, 0.0, self.P_eng_max)
        hi = torch.clamp(P_req - P_b_min, 0.0, self.P_eng_max)
        hi = torch.where(P_req <= 0, torch.zeros_like(hi), hi)   # phanh: tat may
        return lo, torch.maximum(hi, lo)

    def fuel_rate(self, P_eng):
        on = torch.sigmoid((P_eng - 1500.0) / 400.0)  # "may bat" dang mem -> kha vi
        return on * (self.f0 + self.f1 * P_eng + self.f2 * P_eng ** 2)

    def step_detail(self, soc, v, acc, P_eng):
        """Nhu step() nhung tra ve them P_req, P_batt, dong dien I (dung cho benchmark)."""
        P_req = self.power_demand(v, acc)
        P_b_min, P_b_max = self.battery_limits(soc)
        P_batt = torch.clamp(P_req - P_eng, P_b_min, P_b_max)   # du thua khi phanh -> phanh co
        V = self.voc(soc)
        disc = torch.clamp(V ** 2 - 4 * self.R * P_batt, min=1.0)
        I = (V - torch.sqrt(disc)) / (2 * self.R)
        soc_next = soc - I * self.dt / self.Q
        return dict(soc_next=soc_next, fuel=self.fuel_rate(P_eng) * self.dt,
                    P_req=P_req, P_batt=P_batt, I=I, Voc=V)

    def step(self, soc, v, acc, P_eng):
        """f_phys: (SOC, v, a, P_eng) -> (SOC_next, fuel). Phuong trinh pin R_int."""
        d = self.step_detail(soc, v, acc, P_eng)
        return d['soc_next'], d['fuel']


PHYS = HEVPhysics()
SOC_REF, ALPHA, BETA, GAMMA = 0.6, 1.0, 500.0, 0.99


def reward_phys(soc_next, fuel):
    """r = -(alpha*fuel + beta*(SOC_ref - SOC)^2) - giong reward cua repo."""
    return -(ALPHA * fuel + BETA * (SOC_REF - soc_next) ** 2)


# =============================================================================
# 2. NEURAL NETWORKS VOI VAT LY BEN TRONG
# =============================================================================
class PhysicsFeatureLayer(nn.Module):
    """(SOC, v, a) -> [SOC, v, a, P_req, lo, hi] da chuan hoa. Vat ly lam feature."""

    def forward(self, s):
        soc, v, acc = s[:, 0:1], s[:, 1:2], s[:, 2:3]
        P_req = PHYS.power_demand(v, acc)
        lo, hi = PHYS.engine_bounds(soc, v, acc)
        return torch.cat([(soc - SOC_REF) * 10, v / 30, acc / 2,
                          P_req / 5e4, lo / 5e4, hi / 5e4], dim=1)


def mlp(n_in, n_out):
    return nn.Sequential(nn.Linear(n_in, 64), nn.ReLU(),
                         nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, n_out))


class Actor(nn.Module):
    """s -> [PhysicsFeature] -> MLP -> sigmoid -> [PhysicsConstraint] -> P_eng kha thi."""

    def __init__(self):
        super().__init__()
        self.feat = PhysicsFeatureLayer()
        self.net = mlp(6, 1)

    def forward(self, s, noise_std=0.0):
        logit = self.net(self.feat(s))
        if noise_std > 0:                           # kham pha trong khong gian logit
            logit = logit + noise_std * torch.randn_like(logit)
        u = torch.sigmoid(logit)                    # u in (0, 1)
        lo, hi = PHYS.engine_bounds(s[:, 0:1], s[:, 1:2], s[:, 2:3])
        return lo + u * (hi - lo)                   # PhysicsConstraintLayer


class PhysicsCritic(nn.Module):
    """Q(s, a | v', a') = r_phys(s,a) + gamma * V_nn(f_phys(s,a), v', a').

    v', a' la van toc/gia toc buoc sau lay tu chu trinh lai (ngoai sinh).
    """

    def __init__(self, gamma=GAMMA):
        super().__init__()
        self.feat = PhysicsFeatureLayer()
        self.V = mlp(6, 1)
        self.gamma = gamma

    def value(self, s):
        return self.V(self.feat(s))

    def forward(self, s, P_eng, exo_next):
        soc_next, fuel = PHYS.step(s[:, 0:1], s[:, 1:2], s[:, 2:3], P_eng)  # vat ly
        r = reward_phys(soc_next, fuel)                                       # vat ly
        s_next = torch.cat([soc_next, exo_next], dim=1)
        return r + self.gamma * self.value(s_next)                                 # NN


# =============================================================================
# 3. PIRL AGENT
# =============================================================================
class PIRLAgent:
    def __init__(self, lr_a=1e-3, lr_c=1e-3, tau=0.005, buffer=100_000, gamma=GAMMA):
        self.actor, self.critic = Actor(), PhysicsCritic(gamma)
        self.actor_t, self.critic_t = copy.deepcopy(self.actor), copy.deepcopy(self.critic)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=lr_a)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=lr_c)
        self.tau = tau
        # Replay chi can luu (s, v', a'): reward va SOC' tinh lai bang vat ly
        self.buf = np.zeros((buffer, 5), dtype=np.float32)
        self.n, self.cap = 0, buffer

    def store(self, s, exo_next):
        self.buf[self.n % self.cap] = np.concatenate([s, exo_next])
        self.n += 1

    @torch.no_grad()
    def act(self, s, noise_std=0.0):
        return self.actor(torch.as_tensor(s, dtype=torch.float32)[None], noise_std).item()

    def update(self, batch=256):
        idx = np.random.randint(0, min(self.n, self.cap), batch)
        b = torch.as_tensor(self.buf[idx])
        s, exo = b[:, :3], b[:, 3:]

        # --- Critic (V_nn): V(s) <- r_phys(s, pi(s)) + gamma * V_targ(f_phys(s, pi(s)))
        with torch.no_grad():
            y = self.critic_t(s, self.actor_t(s), exo)
        loss_c = ((self.critic.value(s) - y) ** 2).mean()
        self.opt_c.zero_grad(); loss_c.backward(); self.opt_c.step()

        # --- Actor: max Q; dQ/da di qua dr/da va dSOC'/da (gradient vat ly chinh xac)
        loss_a = -self.critic(s, self.actor(s), exo).mean()
        self.opt_a.zero_grad(); loss_a.backward(); self.opt_a.step()

        # --- Soft update target
        for net, tgt in ((self.actor, self.actor_t), (self.critic, self.critic_t)):
            for p, pt in zip(net.parameters(), tgt.parameters()):
                pt.data.mul_(1 - self.tau).add_(self.tau * p.data)
        return loss_c.item(), loss_a.item()


# =============================================================================
# 4. MOI TRUONG + TRAIN + DANH GIA
# =============================================================================
def load_cycle(name):
    v = scio.loadmat(os.path.join(CYCLE_DIR, name + '.mat'))['speed_vector'].ravel()
    v = v.astype(np.float32)
    acc = np.append(np.diff(v), 0.0).astype(np.float32)   # a(t) = v(t+1) - v(t)
    return v, acc


def run_episode(policy, v, acc, soc0=SOC_REF, agent=None, noise_std=0.0):
    """Mo phong mot chu trinh. Moi truong dung chung HEVPhysics (co the thay bang Prius_model)."""
    soc, fuel_total, socs = soc0, 0.0, []
    for t in range(len(v) - 1):
        s = np.array([soc, v[t], acc[t]], dtype=np.float32)
        P_eng = policy(s, noise_std)
        with torch.no_grad():
            soc_next, fuel = PHYS.step(*(torch.tensor([[x]]) for x in (soc, v[t], acc[t], P_eng)))
        if agent is not None:
            agent.store(s, np.array([v[t + 1], acc[t + 1]], dtype=np.float32))
            if agent.n >= 1000:
                agent.update()
        soc = soc_next.item()
        fuel_total += fuel.item()
        socs.append(soc)
    return fuel_total, soc, np.array(socs)


def rule_based(s, _noise=0.0):
    """Baseline: dong co chay 20 kW (vung hieu suat tot) khi SOC < ref va P_req > 0."""
    soc, v, acc = (torch.tensor([[x]]) for x in s)
    lo, hi = PHYS.engine_bounds(soc, v, acc)
    P_req = PHYS.power_demand(v, acc)
    target = torch.where(soc < SOC_REF, torch.clamp(P_req + 5e3, min=2e4), P_req * 0)
    return torch.clamp(target, lo, hi).item()


def soc_corrected_fuel(fuel, soc_end):
    """Quy doi chenh lech SOC cuoi ra nhien lieu de so sanh cong bang."""
    E_batt = (soc_end - SOC_REF) * PHYS.Q * PHYS.voc(torch.tensor(SOC_REF)).item()  # J
    return fuel - E_batt * PHYS.f1 / 0.9


def main(episodes=30):
    agent = PIRLAgent()
    v_tr, a_tr = load_cycle('Standard_UDDS')
    for ep in range(episodes):
        noise = max(0.05, 1.0 * (1 - ep / episodes))
        soc0 = np.random.uniform(0.5, 0.7)
        fuel, soc_end, _ = run_episode(agent.act, v_tr, a_tr, soc0, agent, noise)
        print(f'ep {ep:2d} | noise {noise:.2f} | SOC0 {soc0:.2f} -> {soc_end:.3f} | fuel {fuel:7.1f} g')

    print('\n=== Danh gia (khong noise, SOC0 = 0.6) ===')
    print(f'{"cycle":16s} {"policy":6s} {"fuel(g)":>8s} {"SOC_end":>8s} {"fuel_corr(g)":>12s}')
    results = {}
    for cyc in ['Standard_UDDS', 'Standard_NEDC', 'Standard_WVUSUB']:
        v, acc = load_cycle(cyc)
        for name, pol in (('PIRL', agent.act), ('Rule', rule_based)):
            fuel, soc_end, socs = run_episode(pol, v, acc)
            results[(cyc, name)] = socs
            print(f'{cyc:16s} {name:6s} {fuel:8.1f} {soc_end:8.3f} {soc_corrected_fuel(fuel, soc_end):12.1f}')

    torch.save(agent.actor.state_dict(), os.path.join(HERE, 'pirl_actor.pt'))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        v, _ = load_cycle('Standard_NEDC')
        ax[0].plot(v * 3.6, 'k'); ax[0].set_ylabel('Speed (km/h)')
        for name in ('PIRL', 'Rule'):
            ax[1].plot(results[('Standard_NEDC', name)], label=name)
        ax[1].axhline(SOC_REF, ls='--', c='gray'); ax[1].set_ylabel('SOC')
        ax[1].set_xlabel('Time (s)'); ax[1].legend(); ax[0].set_title('NEDC (test)')
        fig.tight_layout(); fig.savefig(os.path.join(HERE, 'pirl_soc_nedc.png'), dpi=120)
    except ImportError:
        pass


if __name__ == '__main__':
    main()
