
import torch
import util
from ipdb import set_trace as debug

def compute_sb_DSB_train(opt, label, label_aux,dyn, ts, ms, policy_opt, return_z=False, itr=None):
    """ Implementation of Eq (18,19) in our main paper.
    """
    dt      = dyn.dt
    zs = policy_opt(ms,ts)
    g_ts = dyn.g(ts)
    g_ts = g_ts[:,None,None,None] if util.is_image_dataset(opt) else g_ts[:,None]
    reg =  torch.nn.functional.mse_loss(g_ts*dt*zs,label_aux)
    loss = opt.reg*torch.nn.functional.mse_loss(g_ts*dt*zs,label)+(1-opt.reg)*reg
    return loss, zs, reg if return_z else loss
