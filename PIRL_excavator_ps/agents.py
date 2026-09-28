# -*- coding: utf-8 -*-
"""
Cac bo dieu khien cho may xuc hybrid POWER-SPLIT:
  PIRL, DDPG, TD3, SAC (model-free), Rule, A-ECMS, DP (noi long).

State  s = [SOC, T_pin, n_ICE, n_bom, n_mode, P_hyd, P_boom, w_hm]
Action a = (n_bom, n_ICE)
"""
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ps_model import PHYS, reward, LHV, project, S_DIM
from excavator_model import _interp2

NP_MIN, NP_MAX, NE_MIN, NE_MAX = 1000.0, 2200.0, 900.0, 2200.0   # thang action cho agent model-free


def raw_feat(s):
    return torch.stack([(s[:, 0] - PHYS.soc_ref) * 10, (s[:, 1] - 40) / 10, (s[:, 2] - 1600) / 400,
                        (s[:, 3] - 1600) / 400, (s[:, 4] - 1600) / 400, s[:, 5] / 1e5, s[:, 6] / 3e4,
                        (s[:, 6] > 100).float()], dim=1)


def phys_feat(s):
    """PhysicsFeatureLayer: cua so toc do bom, mien toc do ICE, toc do trung tinh (EMG dung yen), gioi han pin."""
    p_lo, p_hi = PHYS.pump_window(s, pump_req=True)
    e_lo, e_hi = PHYS.engine_bounds(s, p_hi)
    b_min, b_max = PHYS.batt_limits(s[:, 0], s[:, 1])
    return torch.cat([raw_feat(s), torch.stack([
        (p_lo - 1600) / 400, (p_hi - 1600) / 400, (e_lo - 1600) / 400, (e_hi - 1600) / 400,
        (PHYS.n_e_neutral(p_hi) - 1600) / 400, b_max / 5e4, b_min / 3e4], dim=1)], dim=1)


N_RAW, N_PHYS = 8, 15


def mlp(n_in, n_out, h=64):
    return nn.Sequential(nn.Linear(n_in, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, n_out))


def to_norm(a):          # (n_bom, n_ICE) -> [0,1]^2 (luu trong replay)
    return np.array([(a[0] - NP_MIN) / (NP_MAX - NP_MIN), (a[1] - NE_MIN) / (NE_MAX - NE_MIN)], dtype=np.float32)


def from_norm(u):        # [...,2] -> n_bom, n_ICE
    return NP_MIN + u[..., 0] * (NP_MAX - NP_MIN), NE_MIN + u[..., 1] * (NE_MAX - NE_MIN)


class Buffer:
    def __init__(self, cap=200_000):
        self.d = np.zeros((cap, 2 * S_DIM + 3), dtype=np.float32)
        self.n, self.cap = 0, cap

    def add(self, s, a, r, s2):
        self.d[self.n % self.cap] = np.concatenate([s, to_norm(a), [r], s2])
        self.n += 1

    def sample(self, b):
        x = torch.as_tensor(self.d[np.random.randint(0, min(self.n, self.cap), b)])
        return x[:, :S_DIM], x[:, S_DIM:S_DIM + 2], x[:, S_DIM + 2], x[:, S_DIM + 3:]


def soft_update(net, tgt, tau):
    with torch.no_grad():
        for p, pt in zip(net.parameters(), tgt.parameters()):
            pt.mul_(1 - tau).add_(tau * p)


class _RLBase:
    warmup, batch = 1000, 256

    def __init__(self):
        self.buf = Buffer()

    def observe(self, s, a_cmd, r, s2):
        self.buf.add(s, a_cmd, r, s2)
        if self.buf.n >= self.warmup:
            self.update()

    def _st(self, s):
        return torch.as_tensor(s, dtype=torch.float32)[None]


# =============================================================================
# PIRL
# =============================================================================
class PIRLActor(nn.Module):
    def __init__(self, feat=True, constr=True):
        super().__init__()
        self.feat, self.constr = feat, constr
        self.net = mlp(N_PHYS if feat else N_RAW, 2)

    def forward(self, s, noise=0.0):
        logit = self.net(phys_feat(s) if self.feat else raw_feat(s))
        if noise > 0:
            logit = logit + noise * torch.randn_like(logit)
        u = torch.sigmoid(logit)
        if not self.constr:
            return from_norm(u)
        # ---- PhysicsConstraintLayer: n_bom trong cua so dat duoc (+ du luu luong),
        #      n_ICE trong mien kha thi (mo-men ICE, toc do EMG, gioi han pin, nhiet) tai n_bom do
        p_lo, p_hi = PHYS.pump_window(s, pump_req=True)
        n_p = p_lo + u[:, 0] * (p_hi - p_lo)
        e_lo, e_hi = PHYS.engine_bounds(s, n_p)
        return n_p, e_lo + u[:, 1] * (e_hi - e_lo)


class PIRLAgent(_RLBase):
    """feat / constr / phys_critic = True la PIRL day du; tat tung cai de lam ablation."""

    def __init__(self, gamma=0.95, feat=True, constr=True, phys_critic=True, lr=1e-3, tau=0.005):
        super().__init__()
        self.gamma, self.tau = gamma, tau
        self.feat, self.constr, self.phys_critic = feat, constr, phys_critic
        self.actor = PIRLActor(feat, constr)
        nf = N_PHYS if feat else N_RAW
        self.critic = mlp(nf if phys_critic else nf + 2, 1)      # V(s) hoac Q(s,a)
        self.actor_t, self.critic_t = copy.deepcopy(self.actor), copy.deepcopy(self.critic)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=lr)

    def _x(self, s):
        return phys_feat(s) if self.feat else raw_feat(s)

    def act(self, s, explore=False, frac=1.0):
        with torch.no_grad():
            noise = max(0.05, 1.0 * (1 - frac)) if explore else 0.0
            n, P = self.actor(self._st(s), noise)
            return n.item(), P.item()

    def q_phys(self, critic, s, n, P, exo):
        """Q = r_phys(s,a) + gamma * V_nn(f_phys(s,a)) - vat ly nam trong critic."""
        if not self.constr:
            n, P = project(s, n, P)
        d = PHYS.step_detail(s, n, P)
        s2 = torch.cat([d['soc'][:, None], d['Tb'][:, None], P[:, None], n[:, None], exo], dim=1)
        return reward(d) + self.gamma * critic(self._x(s2))[:, 0]

    def q_nn(self, critic, s, n, P):
        a = torch.stack([(n - NP_MIN) / (NP_MAX - NP_MIN), (P - NE_MIN) / (NE_MAX - NE_MIN)], 1)
        return critic(torch.cat([self._x(s), a], 1))[:, 0]

    def update(self):
        s, a, r, s2 = self.buf.sample(self.batch)
        exo = s2[:, 4:]
        if self.phys_critic:
            with torch.no_grad():
                y = self.q_phys(self.critic_t, s, *self.actor_t(s), exo)
            loss_c = F.mse_loss(self.critic(self._x(s))[:, 0], y)
        else:
            with torch.no_grad():
                y = r + self.gamma * self.q_nn(self.critic_t, s2, *self.actor_t(s2))
            loss_c = F.mse_loss(self.q_nn(self.critic, s, *from_norm(a)), y)
        self.opt_c.zero_grad(); loss_c.backward(); self.opt_c.step()
        n, P = self.actor(s)
        q = self.q_phys(self.critic, s, n, P, exo) if self.phys_critic else self.q_nn(self.critic, s, n, P)
        self.opt_a.zero_grad(); (-q.mean()).backward(); self.opt_a.step()
        soft_update(self.actor, self.actor_t, self.tau)
        soft_update(self.critic, self.critic_t, self.tau)


# =============================================================================
# Model-free: DDPG, TD3, SAC (mang 256-256, input = state chuan hoa, action u in [0,1]^2)
# =============================================================================
class DDPGAgent(_RLBase):
    def __init__(self, gamma=0.95, lr=1e-3, tau=0.005, td3=False):
        super().__init__()
        self.gamma, self.tau, self.td3, self.it = gamma, tau, td3, 0
        self.actor = nn.Sequential(mlp(N_RAW, 2, 256), nn.Sigmoid())
        self.q1, self.q2 = mlp(N_RAW + 2, 1, 256), mlp(N_RAW + 2, 1, 256)
        self.actor_t, self.q1_t, self.q2_t = (copy.deepcopy(m) for m in (self.actor, self.q1, self.q2))
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_c = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)

    def act(self, s, explore=False, frac=1.0):
        with torch.no_grad():
            u = self.actor(raw_feat(self._st(s)))[0].numpy()
        if explore:
            u = u + max(0.02, 0.3 * (1 - frac)) * np.random.randn(2)
        n, P = from_norm(np.clip(u, 0, 1))
        return float(n), float(P)

    def update(self):
        s, a, r, s2 = self.buf.sample(self.batch)
        x, x2, r = raw_feat(s), raw_feat(s2), r[:, None]
        with torch.no_grad():
            a2 = self.actor_t(x2)
            if self.td3:                                      # target policy smoothing + clipped double-Q
                a2 = torch.clamp(a2 + torch.clamp(0.1 * torch.randn_like(a2), -0.25, 0.25), 0, 1)
                q_t = torch.minimum(self.q1_t(torch.cat([x2, a2], 1)), self.q2_t(torch.cat([x2, a2], 1)))
            else:
                q_t = self.q1_t(torch.cat([x2, a2], 1))
            y = r + self.gamma * q_t
        loss = F.mse_loss(self.q1(torch.cat([x, a], 1)), y)
        if self.td3:
            loss = loss + F.mse_loss(self.q2(torch.cat([x, a], 1)), y)
        self.opt_c.zero_grad(); loss.backward(); self.opt_c.step()
        self.it += 1
        if not self.td3 or self.it % 2 == 0:                  # TD3: cap nhat actor tre
            loss_a = -self.q1(torch.cat([x, self.actor(x)], 1)).mean()
            self.opt_a.zero_grad(); loss_a.backward(); self.opt_a.step()
            soft_update(self.actor, self.actor_t, self.tau)
            soft_update(self.q1, self.q1_t, self.tau)
            soft_update(self.q2, self.q2_t, self.tau)


class TD3Agent(DDPGAgent):
    def __init__(self, gamma=0.95, **kw):
        super().__init__(gamma=gamma, td3=True, **kw)


class SACAgent(_RLBase):
    def __init__(self, gamma=0.95, lr=3e-4, tau=0.005):
        super().__init__()
        self.gamma, self.tau = gamma, tau
        self.pi = mlp(N_RAW, 4, 256)
        self.q1, self.q2 = mlp(N_RAW + 2, 1, 256), mlp(N_RAW + 2, 1, 256)
        self.q1_t, self.q2_t = copy.deepcopy(self.q1), copy.deepcopy(self.q2)
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.opt_pi = torch.optim.Adam(self.pi.parameters(), lr=lr)
        self.opt_q = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)
        self.opt_al = torch.optim.Adam([self.log_alpha], lr=lr)

    def _sample(self, x, deterministic=False):
        mu, log_std = self.pi(x).chunk(2, dim=1)
        log_std = torch.clamp(log_std, -5, 2)
        z = mu if deterministic else mu + log_std.exp() * torch.randn_like(mu)
        t = torch.tanh(z)
        logp = (-0.5 * ((z - mu) / log_std.exp()) ** 2 - log_std - 0.5 * math.log(2 * math.pi)
                - torch.log(1 - t ** 2 + 1e-6) - math.log(0.5)).sum(1, keepdim=True)
        return (t + 1) / 2, logp

    def act(self, s, explore=False, frac=1.0):
        with torch.no_grad():
            u, _ = self._sample(raw_feat(self._st(s)), not explore)
        n, P = from_norm(u[0].numpy())
        return float(n), float(P)

    def update(self):
        s, a, r, s2 = self.buf.sample(self.batch)
        x, x2, r = raw_feat(s), raw_feat(s2), r[:, None]
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            a2, lp2 = self._sample(x2)
            q_t = torch.minimum(self.q1_t(torch.cat([x2, a2], 1)), self.q2_t(torch.cat([x2, a2], 1)))
            y = r + self.gamma * (q_t - alpha * lp2)
        loss_q = F.mse_loss(self.q1(torch.cat([x, a], 1)), y) + F.mse_loss(self.q2(torch.cat([x, a], 1)), y)
        self.opt_q.zero_grad(); loss_q.backward(); self.opt_q.step()
        an, lp = self._sample(x)
        q = torch.minimum(self.q1(torch.cat([x, an], 1)), self.q2(torch.cat([x, an], 1)))
        loss_pi = (alpha * lp - q).mean()
        self.opt_pi.zero_grad(); loss_pi.backward(); self.opt_pi.step()
        loss_al = -(self.log_alpha * (lp.detach() - 2.0)).mean()      # target entropy = -dim(A) = -2
        self.opt_al.zero_grad(); loss_al.backward(); self.opt_al.step()
        soft_update(self.q1, self.q1_t, self.tau)
        soft_update(self.q2, self.q2_t, self.tau)


# =============================================================================
# Rule-based, A-ECMS, DP (dua tren mo hinh)
# =============================================================================
class RulePolicy:
    """EMS luat: bom chay o toc do che do (nhu may thuong); ICE bam toc do trung tinh (EMG ~ dung yen)
    + hieu chinh theo SOC: SOC thap -> ICE nhanh hon -> EMG phat dien."""

    def __init__(self, k_soc=4000.0, offset=0.0, phys=PHYS):
        self.k, self.off, self.phys = k_soc, offset, phys

    def reset(self):
        pass

    def __call__(self, s):
        with torch.no_grad():
            st = torch.as_tensor(s, dtype=torch.float64)[None]
            _, p_hi = self.phys.pump_window(st, pump_req=True)
            n_e = self.phys.n_e_neutral(p_hi) + self.off + self.k * (self.phys.soc_ref - s[0])
            return p_hi.item(), n_e.item()


class ECMSPolicy:
    """A-ECMS: min [fuel + W_age*age + W_unmet*unmet + s_eq * EQ * E_pin], s_eq = s0 + kp (SOC_ref - SOC).
    Luoi 7 toc do bom x 11 toc do ICE (trong mien kha thi) moi buoc."""

    def __init__(self, s0=1.0, kp=500.0, n_p=7, n_e=11, phys=PHYS):
        self.s0, self.kp, self.phys = s0, kp, phys
        g = torch.meshgrid(torch.linspace(0, 1, n_p, dtype=torch.float64),
                           torch.linspace(0, 1, n_e, dtype=torch.float64), indexing='ij')
        self.up, self.ue = g[0].reshape(-1), g[1].reshape(-1)
        self.eq = 1.0 / (0.40 * LHV)

    def reset(self):
        pass

    def __call__(self, s):
        with torch.no_grad():
            st = torch.as_tensor(s, dtype=torch.float64)[None].expand(len(self.up), S_DIM)
            p_lo, p_hi = self.phys.pump_window(st, pump_req=True)
            n_p = p_lo + self.up * (p_hi - p_lo)
            e_lo, e_hi = self.phys.engine_bounds(st, n_p)
            n_e = e_lo + self.ue * (e_hi - e_lo)
            d = self.phys.step_detail(st, n_p, n_e)
            s_eq = self.s0 + self.kp * (self.phys.soc_ref - s[0])
            J = d['fuel'] + self.phys.W_AGE * d['age_Ah'] + self.phys.W_UNMET * d['unmet'] \
                + s_eq * self.eq * d['V'] * d['I'] * self.phys.dt
            i = J.argmin()
            return n_p[i].item(), n_e[i].item()


def relaxed(phys):
    """Ban sao mo hinh bo gioi han toc do doi va quan tinh -> DP tren no cho CAN DUOI."""
    import copy as _c
    p = _c.copy(phys)
    p.dn_max, p.J_e, p.J_p, p.J_m = 1e4, 0.0, 0.0, 0.0
    return p


class DPPolicy:
    """DP 2 chieu (SOC x T_pin) tren mo hinh NOI LONG (khong gioi han toc do doi, khong quan tinh):
    action = luoi tuyet doi 7 toc do bom x 14 toc do ICE. Chay tren chinh mo hinh noi long -> can duoi."""

    def __init__(self, cyc, phys, n_soc=61, n_T=5, T_rng=(32.0, 50.0), w_term=1e6):
        self.phys, self.k = phys, 0
        f = torch.float64
        self.sg = torch.linspace(phys.soc_min, phys.soc_max, n_soc, dtype=f)
        self.tg = torch.linspace(*T_rng, n_T, dtype=f)
        self.ue = torch.arange(900.0, 2201.0, 100.0, dtype=f)
        self.up = torch.linspace(0, 1, 7, dtype=f)
        self.exo = torch.as_tensor(np.stack([cyc['n'], cyc['P_hyd'], cyc['P_boom'], cyc['w_hm']], 1), dtype=f)
        N = len(self.exo)
        S, T = torch.meshgrid(self.sg, self.tg, indexing='ij')
        self.J = [None] * (N + 1)
        self.J[N] = w_term * (S - phys.soc_ref) ** 2
        with torch.no_grad():
            for k in range(N - 1, -1, -1):
                self.J[k] = self._q(S[..., None, None], T[..., None, None], k)[0].flatten(-2).min(-1).values

    def _interp(self, J, soc, Tb):
        return _interp2(float(self.sg[0]), float(self.sg[1] - self.sg[0]),
                        float(self.tg[0]), float(self.tg[1] - self.tg[0]), J, soc, Tb)

    def _q(self, S, T, k):
        e = self.exo[k]
        shape = torch.broadcast_shapes(S.shape[:-2] + (len(self.up), len(self.ue)))
        lo = torch.clamp(torch.maximum(e[1] / self.phys.k_pump, torch.tensor(self.phys.np_min, dtype=S.dtype)),
                         max=float(torch.clamp(e[0], self.phys.np_min, self.phys.np_max)))
        hi = torch.clamp(e[0], self.phys.np_min, self.phys.np_max)
        n_p = (lo + self.up * (hi - lo))[:, None].expand(shape)
        n_e = self.ue.expand(shape)
        st = torch.stack([S.expand(shape), T.expand(shape), n_e, n_p] + [e[i].expand(shape) for i in range(4)], -1)
        d = self.phys.step_detail(st, n_p, n_e)
        cost = d['fuel'] + self.phys.W_AGE * d['age_Ah'] + self.phys.W_UNMET * d['unmet']
        return cost + self._interp(self.J[k + 1], d['soc'], d['Tb']), n_p, n_e

    def reset(self):
        self.k = 0

    def __call__(self, s):
        with torch.no_grad():
            S = torch.tensor([[float(s[0])]], dtype=torch.float64)
            T = torch.tensor([[float(s[1])]], dtype=torch.float64)
            q, n_p, n_e = self._q(S[..., None, None], T[..., None, None], self.k)
            self.k += 1
            i = q.flatten().argmin()
            return n_p.flatten()[i].item(), n_e.flatten()[i].item()
