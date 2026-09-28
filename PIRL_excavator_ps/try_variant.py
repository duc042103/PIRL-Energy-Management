# Thu nhanh bien the chong bao hoa: python try_variant.py <seed> <reg> <qnorm 0/1>
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, numpy as np
import bench as B, agents as A
sd, reg, qn = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
torch.manual_seed(sd); np.random.seed(sd); torch.set_num_threads(1)
ag = A.PIRLAgent(0.99); ag.logit_reg = reg
if not qn:
    import types
    def upd(self):
        import torch.nn.functional as F
        s, a, r, s2 = self.buf.sample(self.batch); exo = s2[:, 4:]
        with torch.no_grad():
            y = self.q_phys(self.critic_t, s, *self.actor_t(s), exo)
        lc = F.mse_loss(self.critic(self._x(s))[:, 0], y)
        self.opt_c.zero_grad(); lc.backward(); self.opt_c.step()
        n, P = self.actor(s); q = self.q_phys(self.critic, s, n, P, exo)
        la = -q.mean() + self.logit_reg * (self.actor.last_logit ** 2).mean()
        self.opt_a.zero_grad(); la.backward(); torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0); self.opt_a.step()
        A.soft_update(self.actor, self.actor_t, self.tau); A.soft_update(self.critic, self.critic_t, self.tau)
    ag.update = types.MethodType(upd, ag)
val = B.make_cycle('train_mixed', 999)
for ep in range(10):
    cyc = B.make_cycle('train_mixed', seed=100 + ep + 1000 * sd)
    B.simulate(lambda s: ag.act(s, True, ep / 30), cyc, np.random.uniform(0.45, 0.65), np.random.uniform(33, 45), on_step=ag.observe)
    m = B.metrics(B.simulate(lambda s: ag.act(s), val))
    print(f"s{sd} reg{reg} qn{qn} ep {ep} cost {m['cost_usd_h']:.3f} soc {m['soc_end']:.3f} ret {m['return']:.0f}", flush=True)
