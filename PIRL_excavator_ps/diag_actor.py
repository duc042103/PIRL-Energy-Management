# -*- coding: utf-8 -*-
"""Chan doan: actor PIRL co bi bao hoa (sigmoid ~0/1 -> gradient ~0) khong."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import bench as B
import agents as A

cyc = B.make_cycle('train_mixed', 999)
for sd in [1, 2, 3]:
    ag = A.PIRLAgent(0.99)
    ag.actor.load_state_dict(torch.load(os.path.join(B.RUN_DIR, f'pirl_g0.99_s{sd}.pt')))
    S = []
    B.simulate(lambda s: (S.append(s), ag.act(s))[1], cyc)
    S = torch.as_tensor(np.array(S))
    with torch.no_grad():
        lg = ag.actor.net(A.phys_feat(S))
    u = torch.sigmoid(lg)
    low = S[:, 6] > 100
    print(sd, 'logit mean', lg.mean(0).numpy().round(1), 'abs max', lg.abs().max(0).values.numpy().round(1),
          'sat frac', ((u < 0.02) | (u > 0.98)).float().mean(0).numpy().round(2),
          'u_e M1 %.2f M2 %.2f' % (u[~low, 1].mean().item(), u[low, 1].mean().item()), flush=True)
