
import os, time, gc
from models.MIOFlow.models import make_model, Autoencoder
import numpy as np

import torch
import torch.nn.functional as F
from torch.optim import SGD, RMSprop, Adagrad, AdamW, lr_scheduler, Adam
from torch.utils.tensorboard import SummaryWriter
from torch_ema import ExponentialMovingAverage
from metrics import MMD_loss,compute_metrics,metric_build
import policy
import sde
from loss import compute_sb_DSB_train
import data
import util

from ipdb import set_trace as debug

def build_optimizer_ema_sched(opt, policy):
    direction = policy.direction

    optim_name = {
        'Adam': Adam,
        'AdamW': AdamW,
        'Adagrad': Adagrad,
        'RMSprop': RMSprop,
        'SGD': SGD,
    }.get(opt.optimizer)

    optim_dict = {
            "lr": opt.lr_f if direction=='forward' else opt.lr_b,
            'weight_decay':opt.l2_norm,
    }
    if opt.optimizer == 'SGD':
        optim_dict['momentum'] = 0.9

    optimizer   = optim_name(policy.parameters(), **optim_dict)
    ema         = ExponentialMovingAverage(policy.parameters(), decay=0.999)
    if opt.lr_gamma < 1.0:
        sched = lr_scheduler.StepLR(optimizer, step_size=opt.lr_step, gamma=opt.lr_gamma)
    else:
        sched = None

    return optimizer, ema, sched

def freeze_policy(policy):
    for p in policy.parameters():
        p.requires_grad = False
    policy.eval()
    return policy

def activate_policy(policy):
    for p in policy.parameters():
        p.requires_grad = True
    policy.train()
    return policy

class Runner():
    def __init__(self,opt):
        super(Runner,self).__init__()

        self.start_time = time.time()
        self.ts         = torch.linspace(opt.t0, opt.T, opt.interval)
        
        if opt.use_ae:
            opt.ae=sde.setup_ae(opt)

        self.x_dists    = data.build(opt)
        if opt.problem_name == 'petal' or opt.problem_name =='RNAsc':
            self.x_data = [dist.ground_truth for dist in self.x_dists]

        self.v_dists    = {dist:opt.v_scale*torch.randn(opt.samp_bs, *opt.data_dim) for dist in range(len(self.x_dists))}                    
        # Build metrics
        self.metrics    = metric_build(opt)
        # build dynamics, forward (z_f) and backward (z_b) policies
        self.dyn        = sde.build(opt, self.x_dists, self.v_dists)
        self.z_f        = policy.build(opt, self.dyn, 'forward')  # p -> q
        self.z_b        = policy.build(opt, self.dyn, 'backward') # q -> p

        self.optimizer_f, self.ema_f, self.sched_f = build_optimizer_ema_sched(opt, self.z_f)
        self.optimizer_b, self.ema_b, self.sched_b = build_optimizer_ema_sched(opt, self.z_b)


        if opt.load:
            util.restore_checkpoint(opt, self, opt.load)
            self.dyn.prev_v_boundary = self.v_dists

        if opt.log_tb: # tensorboard related things
            self.it_f   = 0
            self.it_b   = 0
            self.writer =SummaryWriter(
                log_dir =os.path.join('runs', opt.dir)
            )

    def update_count(self, direction):
        if direction == 'forward':
            self.it_f += 1
            return self.it_f
        elif direction == 'backward':
            self.it_b += 1
            return self.it_b
        else:
            raise RuntimeError()

    def get_optimizer_ema_sched(self, z):
        if z == self.z_f:
            return self.optimizer_f, self.ema_f, self.sched_f
        elif z == self.z_b:
            return self.optimizer_b, self.ema_b, self.sched_b
        else:
            raise RuntimeError()

    @torch.no_grad()
    def sample_train_data(self, opt, policy_opt, policy_impt, reused_sampler, rollout=None, resample=None):

        # reuse or sample training ms and zs
        try:
            reused_traj         = next(reused_sampler)
            train_ms, train_zs  = reused_traj[:,0,...], reused_traj[:,1,...]
            print('generate train data from [{}]!'.format(util.green('reused samper')))
        except:
            _, ema, _           = self.get_optimizer_ema_sched(policy_opt)
            _, ema_impt, _      = self.get_optimizer_ema_sched(policy_impt)
            with ema.average_parameters(), ema_impt.average_parameters():
                policy_impt     = freeze_policy(policy_impt)
                policy_opt      = freeze_policy(policy_opt)

                corrector = (lambda x,t: policy_impt(x,t) + policy_opt(x,t)) if opt.use_corrector else None
                ms, zs, _, labels, ts = self.dyn.sample_traj(self.ts, policy_impt, corrector=corrector, rollout=rollout, resample=resample)
                train_ms        = ms.detach().cpu(); del ms
                train_zs        = zs.detach().cpu(); del zs
                train_labels    = labels.detach().cpu(); del labels
                train_ts        = ts.detach().cpu(); del ts
            
            print('generate train data from [{}]!'.format(util.red('sampling')))
        assert train_ms.shape[0] == opt.samp_bs
        assert train_ms.shape[1] == len(train_ts)
        gc.collect()

        return train_ms, train_zs, train_ts, train_labels

    def sb_alternate_train_stage(self, opt, stage, epoch, direction, reused_sampler=None, rollout=False, resample=True):
        policy_opt, policy_impt = {
            'forward':  [self.z_f, self.z_b], # train forwad,   sample from backward
            'backward': [self.z_b, self.z_f], # train backward, sample from forward
        }.get(direction)

        for ep in range(epoch):
            # prepare training data
            train_ms, train_zs, train_ts, train_labels = self.sample_train_data(
                opt, policy_opt, policy_impt, reused_sampler, rollout=rollout, resample=resample
            )
            # train one epoch
            policy_impt = freeze_policy(policy_impt)
            policy_opt = activate_policy(policy_opt)
            self.DSB_alternate_train_ep(
                opt, ep, stage, direction, train_ms, train_zs, train_ts, train_labels, policy_opt, epoch
            )
    def DSB_alternate_train_ep(
        self, opt, ep, stage, direction, train_xs, train_zs, train_ts, train_labels, policy, num_epoch
    ):
        assert train_xs.shape[0] == opt.samp_bs
        assert train_zs.shape[0] == opt.samp_bs
        assert direction == policy.direction

        optimizer, ema, sched = self.get_optimizer_ema_sched(policy)
        use_amp=opt.use_amp
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        for it in range(opt.num_itr):
            # -------- sample x_idx and t_idx \in [0, interval] --------
            samp_m_idx = torch.randint(opt.samp_bs,  (opt.train_bs_x,),device='cpu')
            samp_t_idx = util.time_sample(opt.interval, policy.direction, opt.train_bs_t)
            if opt.use_arange_t: samp_t_idx = util.time_arange(train_ts.shape[0], policy.direction)
            
            # -------- build sample --------
            sign=1 if policy.direction=='forward' else -1
            ts          = train_ts[samp_t_idx].detach().to(opt.device)
            ms          = train_xs[samp_m_idx][:, samp_t_idx, ...].to(opt.device)
            zs_impt     = train_zs[samp_m_idx][:, samp_t_idx+sign, ...].to(opt.device)
            train_label = train_labels[samp_m_idx][:, samp_t_idx+sign, ...].to(opt.device)
            optimizer.zero_grad(set_to_none=True)

            # -------- handle for batch_x and batch_t ---------
            # (batch, T, xdim) --> (batch*T, xdim)
            ms          = util.flatten_dim01(ms)
            zs_impt     = util.flatten_dim01(zs_impt)
            train_label = util.flatten_dim01(train_label)
            ts = ts.repeat(opt.train_bs_x)
            assert ms.shape[0] == ts.shape[0]
            assert zs_impt.shape[0] == ts.shape[0]

            # -------- compute loss and backprop --------
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss, zs, reg = compute_sb_DSB_train(
                    opt, train_label, zs_impt,self.dyn, ts, ms, policy, return_z=True,itr=it
                )
            assert not torch.isnan(loss)
            
            scaler.scale(loss).backward()

            if opt.grad_clip is not None:
                torch.nn.utils.clip_grad_norm(policy.parameters(), opt.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.step()
            ema.update()
            if sched is not None: sched.step()

            # -------- logging --------
            zs = util.unflatten_dim01(zs, [len(samp_m_idx), len(samp_t_idx)])
            zs_impt = zs_impt.reshape(zs.shape)
            self.log_sb_alternate_train(
                opt, it, ep, stage, loss, reg, zs, zs_impt, optimizer, direction, num_epoch
            )

    def sb_alternate_train(self, opt):
        reused_sampler = self.evaluate(opt, 0, rollout = [0,opt.num_dist-1], resample=False,ode_samp=False)
        bridge_ep = boundry_ep = opt.num_epoch
        if opt.problem_name =='petal': bridge_ep = 1 #Special handle for petal. the distance between distributions are too close.
        for stage in range(opt.num_stage):
            self.sb_alternate_train_stage(
                opt, stage, boundry_ep, 'backward', rollout = [0,opt.num_dist-1], resample=True
            )
            self.sb_alternate_train_stage(
                opt, stage, boundry_ep, 'forward', rollout = [0,opt.num_dist-1], resample=True
            )

            self.sb_alternate_train_stage(
                opt, stage, bridge_ep, 'backward', rollout = [0,opt.num_dist-1], resample=False
            )
            self.sb_alternate_train_stage(
                opt, stage, boundry_ep, 'forward', rollout = [0,opt.num_dist-1], resample=True
            )
            
            self.sb_alternate_train_stage(
                opt, stage, boundry_ep, 'backward', rollout = [0,opt.num_dist-1], resample=True
            )

            self.sb_alternate_train_stage(
                opt, stage, bridge_ep, 'forward', rollout = [0,opt.num_dist-1], resample=False
            )
            reused_sampler = self.evaluate(opt, stage+1, rollout = [0,opt.num_dist-1],resample=False)

        if opt.log_tb: self.writer.close()

    # @torch.no_grad()
    # def _generate_samples_and_reused_trajs(self, opt, batch, n_samples, n_trajs):
    #     assert n_trajs <= n_samples

    #     ts = self.ts
    #     xTs = torch.empty((n_samples, *opt.data_dim), device='cpu')
    #     if n_trajs > 0:
    #         trajs = torch.empty((n_trajs, 2, len(ts), *opt.data_dim), device='cpu')
    #     else:
    #         trajs = None

    #     with self.ema_f.average_parameters(), self.ema_b.average_parameters():
    #         self.z_f = freeze_policy(self.z_f)
    #         self.z_b = freeze_policy(self.z_b)
    #         corrector = (lambda x,t: self.z_f(x,t) + self.z_b(x,t)) if opt.use_corrector else None

    #         it = 0
    #         while it < n_samples:
    #             # sample backward trajs; save traj if needed
    #             save_traj = (trajs is not None) and it < n_trajs
    #             _xs, _zs, _x_T, _ = self.dyn.sample_traj(
    #                 ts, self.z_b, corrector=corrector, save_traj=save_traj)

    #             # fill xTs (for FID usage) and trajs (for training log_q)
    #             xTs[it:it+batch,...] = _x_T.detach().cpu()[0:min(batch,xTs.shape[0]-it),...]
    #             if save_traj:
    #                 trajs[it:it+batch,0,...] = _xs.detach().cpu()[0:min(batch,trajs.shape[0]-it),...]
    #                 trajs[it:it+batch,1,...] = _zs.detach().cpu()[0:min(batch,trajs.shape[0]-it),...]

    #             it += batch

    #     return xTs, trajs

    @torch.no_grad()
    def compute_NLL(self, opt):
        num_NLL_sample = self.p.num_sample
        assert util.is_image_dataset(opt) and num_NLL_sample%opt.samp_bs==0
        bpds=[]
        with self.ema_f.average_parameters(), self.ema_b.average_parameters():
            for _ in range(int(num_NLL_sample/opt.samp_bs)):
                bits_per_dim = self.dyn.compute_nll(opt.samp_bs, self.ts, self.z_f, self.z_b)
                bpds.append(bits_per_dim.detach().cpu().numpy())

        print(util.yellow("=================== NLL={} ======================").format(np.array(bpds).mean()))

    # @torch.no_grad()
    # def evaluate_img_dataset(self, opt, stage, n_reused_trajs=0, metrics=None, rollout=None,):
    #     assert util.is_image_dataset(opt)

    #     fid, snapshot, ckpt = util.evaluate_stage(opt, stage, metrics)

    #     if ckpt:
    #         keys = ['z_f','optimizer_f','ema_f','z_b','optimizer_b','ema_b']
    #         util.save_checkpoint(opt, self, keys, stage)

    #     # return if no evaluation effort needed in this stage
    #     if not (fid or snapshot): return

    #     # either fid or snapshot requires generating sample (meanwhile
    #     # we can collect trajectories and reuse them in later training)
    #     batch = opt.samp_bs
    #     n_reused_trajs = min(n_reused_trajs, opt.num_FID_sample)
    #     n_reused_trajs -= (n_reused_trajs % batch) # make sure n_reused_trajs is divisible by samp_bs
    #     xTs, trajs = self._generate_samples_and_reused_trajs(
    #         opt, batch, opt.num_FID_sample, n_reused_trajs,
    #     )

    #     if fid and util.exist_FID_ckpt(opt):
    #         FID = util.compute_fid(opt, xTs)
    #         print(util.yellow("===================FID={}===============================").format(FID))
    #         if opt.log_tb: self.log_tb(stage, FID, 'FID', 'eval')
    #     else:
    #         print(util.red("Does not exist FID ckpt, please compute FID manually."))

    #     if snapshot:
    #         util.snapshot(opt, xTs, stage, 'backward')

    #     gc.collect()

    #     if trajs is not None:
    #         trajs = trajs.reshape(-1, batch, *trajs.shape[1:])
    #         return util.create_traj_sampler(trajs)

    @torch.no_grad()
    def evaluate(self, opt, stage, rollout=None, resample=False, ode_samp=False):
        corrector = (lambda x,t: self.z_f(x,t) + self.z_b(x,t)) if opt.use_corrector else None
        ODE_drift = (lambda x,t: 0.5*(self.z_b(x,t) - self.z_f(x,t))) if ode_samp else None
        # if util.is_image_dataset(opt):
        #     return self.evaluate_img_dataset(
        #         opt, stage, n_reused_trajs=n_reused_trajs, metrics=metrics
        #     )

        # elif util.is_toy_dataset(opt):
        snapshot, ckpt = util.evaluate_stage(opt, stage)
        snapshot=True
        if ckpt:
            self.v_dists = self.dyn.prev_v_boundary
            keys = ['z_f','optimizer_f','ema_f','z_b','optimizer_b','ema_b','v_dists']
            util.save_checkpoint(opt, self, keys, stage)
        if snapshot:
            print(util.blue('======Ploting visualization image======'))
            for z in [self.z_b, self.z_f]:
                z = freeze_policy(z)
                ms, _, _, _,_ = self.dyn.sample_traj(
                                                self.ts, 
                                                z, 
                                                save_traj=True,
                                                corrector=corrector,
                                                rollout=rollout, 
                                                resample=resample,
                                                test=True, 
                                                ode_drift= ODE_drift
                                                )

                fn = "{}/xs-stage{}-{}".format(z.direction, stage,z.direction)
                if opt.problem_name =='semicircle':
                    util.save_toy_traj(
                        opt, fn, ms.detach().cpu().numpy(), n_snapshot=5, direction=z.direction
                    )
                elif opt.problem_name == 'petal':
                    util.save_petal_traj(
                        opt, fn, ms.detach().cpu().numpy(), n_snapshot=5, direction=z.direction
                    )               
                
                elif opt.problem_name =='gmm':
                    util.save_toy_seg_traj(
                        opt, fn, ms.detach().cpu().numpy(), n_snapshot=5, direction=z.direction
                    )
                elif opt.problem_name =='RNAsc' and z.direction=='forward':
                    processed_data = compute_metrics(opt, ms.detach().cpu().numpy(), self.x_data, self.metrics, self, stage)
                    util.save_PCA_traj2(opt,fn,processed_data,self.x_data)
                        # print(util.blue('======Evaluating 1-Wasserstein distance full======'))
                        # datas=[]
                        # avg_W1  = 0
                        # avg_mmd = 0
                        # for idx, dist in enumerate(self.x_dists):
                        #     W1,data = compute_W1_full(opt,ms.detach().cpu().numpy(), dist.ground_truth,idx, return_x=True)
                        #     MMD     = compute_MMD(self.mmd, opt,ms.detach().cpu().numpy(), dist.ground_truth,idx)
                        #     if idx!=0:
                        #         avg_W1  += W1
                        #         avg_mmd += MMD
                        #     datas.append(data)
                        #     print(util.red('W1 for time{} is {}'.format(idx,W1)))
                        #     print(util.green('MMD for time{} is {}'.format(idx,MMD)))
                        #     self.log_tb(stage, W1, 'W1_full_t{}'.format(idx), 'SB_'+z.direction) 
                        #     self.log_tb(stage, MMD, 'MMD_full_t{}'.format(idx), 'SB_'+z.direction) 

                        # self.log_tb(stage, avg_W1/(len(self.x_dists)-1), 'W1_avg', 'SB_'+z.direction) 
                        # self.log_tb(stage, avg_mmd/(len(self.x_dists)-1), 'MMD_avg', 'SB_'+z.direction) 
                        # gts     = [item.ground_truth for item in self.x_dists]
                        # util.save_PCA_traj2(opt,fn,datas,gts)
                        # print('AVERAGE W1 IS {}'.format(avg_W1/(len(self.x_dists)-1)))
                        # print('AVERAGE MMD IS {}'.format(avg_mmd/(len(self.x_dists)-1)))
                        # print(util.blue('======Done======'))
                    
                    # elif opt.problem_name =='gmm' or opt.problem_name =='petal':
                    #     util.save_toy_seg_traj(
                    #         opt, fn, ms.detach().cpu().numpy(), n_snapshot=5, direction=z.direction
                    #     )

                # print(util.blue('======Done======'))
            
            # if opt.problem_name == 'RNAsc':
            #     print(util.blue('======Evaluating 2-Wasserstein distance======'))
            #     for z in [self.z_f, self.z_b]:
            #         direction   = z.direction
            #         sign        = 1 if direction =='forward' else 0
            #         bound_x     = self.x_dists[0].sample() if direction=='forward' else self.x_dists[-1].sample()
            #         bound_x     = bound_x.detach().cpu().numpy()
            #         mix_traj    = []
            #         avg_val     = 0
            #         avg_mmd     = 0
            #         fn = "{}/xs-stage{}-{}".format(z.direction, stage,z.direction)
            #         for idx,ro in enumerate([[0,1],[1,2],[2,3],[3,4]]):
            #             pred_x  = self.dyn.sample_test(self.ts, z, ro)
            #             pred_x  = pred_x.detach().cpu().numpy()
            #             mix_traj.append(pred_x[0:opt.samp_bs])
            #             ground_truth = self.x_dists[ro[sign]].ground_truth
            #             size    = ground_truth.shape[0]
            #             pred    = pred_x[0:size,...]
            #             W2      = compute_emd2(pred, ground_truth) 
            #             MMD     = self.mmd(torch.Tensor(pred), torch.Tensor(ground_truth))
            #             print(util.red('W1 for time{} is {}'.format(idx,W2)))
            #             print(util.green('MMD for time{} is {}'.format(idx,MMD)))
            #             avg_val+=W2
            #             avg_mmd+=MMD
            #             self.log_tb(stage, W2, 'W2_t{}'.format(ro[sign]), 'SB_'+direction) 
            #         print('AVERAGE W1 IS {}'.format(avg_val/(len(self.x_dists)-1)))
            #         print('AVERAGE MMD IS {}'.format(avg_mmd/(len(self.x_dists)-1)))
            #         # self.log_tb(stage, avg_val/(len(self.x_dists)-1), 'W2_avg{}', 'SB_'+direction) 
            #         # mix_traj = np.concatenate([bound_x]+mix_traj,axis=0) if direction =='forward' else np.concatenate(mix_traj+[bound_x],axis=0)
            #         # util.save_PCA_traj(opt, fn, None, n_snapshot=5, direction=z.direction, sampled_data=mix_traj) 

                print(util.blue('======Done======'))

    def _print_train_itr(self, it, loss, optimizer, num_itr, name):
        time_elapsed = util.get_time(time.time()-self.start_time)
        lr = optimizer.param_groups[0]['lr']
        print("[{0}] train_it {1}/{2} | lr:{3} | loss:{4} | time:{5}"
            .format(
                util.magenta(name),
                util.cyan("{}".format(1+it)),
                num_itr,
                util.yellow("{:.2e}".format(lr)),
                util.red("{:.4f}".format(loss.item())),
                util.green("{0}:{1:02d}:{2:05.2f}".format(*time_elapsed)),
        ))


    def log_sb_alternate_train(self, opt, it, ep, stage, loss, reg, zs, zs_impt, optimizer, direction, num_epoch):
        time_elapsed = util.get_time(time.time()-self.start_time)
        lr = optimizer.param_groups[0]['lr']
        print("[{0}] stage {1}/{2} | ep {3}/{4} | train_it {5}/{6} | lr:{7} | loss:{8} | time:{9}"
            .format(
                util.magenta("SB {}".format(direction)),
                util.cyan("{}".format(1+stage)),
                opt.num_stage,
                util.cyan("{}".format(1+ep)),
                num_epoch,
                util.cyan("{}".format(1+it+opt.num_itr*ep)),
                opt.num_itr*num_epoch,
                util.yellow("{:.2e}".format(lr)),
                util.red("{:+.4f}".format(loss.item())),
                util.green("{0}:{1:02d}:{2:05.2f}".format(*time_elapsed)),
        ))
        if opt.log_tb:
            step = self.update_count(direction)
            neg_elbo = loss + util.compute_z_norm(zs_impt, self.dyn.dt)
            self.log_tb(step, loss.detach(), 'loss', 'SB_'+direction) 
            self.log_tb(step, neg_elbo.detach(), 'neg_elbo', 'SB_'+direction)
            self.log_tb(step, reg.detach(), 'reg', 'SB_'+direction) 
            # if direction == 'forward':
            #     z_norm = util.compute_z_norm(zs, self.dyn.dt)
            #     self.log_tb(step, z_norm.detach(), 'z_norm', 'SB_forward')

    def log_tb(self, step, val, name, tag):
        self.writer.add_scalar(os.path.join(tag,name), val, global_step=step)

