import sys
import os
try:
    import pickle5 as pickle
except:
    import pickle
sys.path.insert(0, './..')

import argparse
import time

import numpy as np

from kayser_model_2d_ssn import Model

parser = argparse.ArgumentParser()
parser.add_argument('--n_e', '-ne', help='number of excitatory cells',type=int, default=1)
parser.add_argument('--n_i', '-ni', help='number of inhibitory cells',type=int, default=1)
parser.add_argument('--init_iter', '-iit', help='initial iteration number',type=int, default=0)
parser.add_argument('--batch_iter', '-bit', help='number of iterations to run per batch',type=int, default=100)
parser.add_argument('--max_iter', '-mit', help='max iteration number',type=int, default=100)
parser.add_argument('--seed', '-s', help='seed',type=int, default=0)
parser.add_argument('--s_x', '-sx', help='feedforward arbor decay length',type=float, default=0.08)
parser.add_argument('--s_s', '-ss', help='retinotopic scatter decay length',type=float, default=0.08)
parser.add_argument('--gain_i', '-gi', help='gain of inhibitory cells',type=float, default=1.0)
parser.add_argument('--wff_sum', '-w', help='feedforward weight strength',type=float, default=1.0)
# parser.add_argument('--hebb_wei', '-hei', help='whether wei has Hebbian learning rule',type=int, default=0)
# parser.add_argument('--hebb_wii', '-hii', help='whether wii has Hebbian learning rule',type=int, default=0)
parser.add_argument('--prune', '-p', help='whether to prune feedforward weights',type=int, default=1)
parser.add_argument('--rec_plast', '-r', help='whether recurrent weights are plastic',type=int, default=1)
parser.add_argument('--rec_i_ltd', '-d', help='factor for inhibitory LTD term',type=float, default=1.0)
parser.add_argument('--n_wave', '-nw', help='number of geniculate waves',type=int, default=15)
parser.add_argument('--n_stim', '-ns', help='number of light/dark sweeping bars',type=int, default=2)
parser.add_argument('--n_shrink', '-nh', help='factor by which to shrink stimuli',type=float, default=1.0)
parser.add_argument('--n_grid', '-ng', help='number of points per grid edge',type=int, default=20)
parser.add_argument('--mode', '-m', help='mode',type=str, default='spont_vis')
parser.add_argument('--test', '-t', help='test?',type=int, default=0)
args = vars(parser.parse_args())
print(args)
n_e = int(args['n_e'])
n_i = int(args['n_i'])
init_iter = int(args['init_iter'])
batch_iter = int(args['batch_iter'])
max_iter = int(args['max_iter'])
seed = int(args['seed'])
s_x = args['s_x']
s_s = args['s_s']
gain_i = args['gain_i']
wff_sum = args['wff_sum']
# hebb_wei = int(args['hebb_wei']) > 0
# hebb_wii = int(args['hebb_wii']) > 0
prune = int(args['prune']) > 0
rec_plast = int(args['rec_plast']) > 0
rec_i_ltd = args['rec_i_ltd']
n_wave = int(args['n_wave'])
n_stim = int(args['n_stim'])
n_shrink = args['n_shrink']
n_grid = int(args['n_grid'])
mode = str(args['mode'])
test = int(args['test']) > 0

n_batch = 36#26 # number of batches to collect weight changes before adjusting weights
dt_stim = 0.1#0.25 # simulation time between stimuli

max_spike_file = 50 # total number of lgn spike count files

# Define where to save results
res_dir = './../results/'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

if test:
    res_dir = res_dir + 'sim_2d_ssn_lgn_{:s}_rfs_ne={:d}_ni={:d}/'.format(mode,n_e,n_i)
else:
    res_dir = res_dir + 'sim_2d_ssn_lgn_{:s}_rfs_ng={:d}_ne={:d}_ni={:d}_sx={:.2f}_ss={:.2f}_gi={:.1f}_w={:.1f}_p={:d}_r={:d}_d={:.1f}/'.format(
        mode,n_grid,n_e,n_i,s_x,s_s,gain_i,wff_sum,prune,rec_plast,rec_i_ltd)
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

# Define where geniculate wave spikes are saved
lgn_dir = './../results/' + '2d_lgn_{:s}_spikes_nw={:d}_ns={:d}_nh={:.2f}_ng={:d}/'.format(mode,n_wave,n_stim,n_shrink,n_grid)
if not os.path.exists(res_dir):
    os.makedirs(res_dir)
    
# sig2 = 0.00095
# s_n = 0.46638972 * np.sqrt(sig2)
# s_b = 3.1936202 * np.sqrt(sig2)
# broad_frac_e = 0.74130374
# broad_frac_i = 0.5984383
# log10JEE = -0.7561108
# log10JEI = -1.1950144
# log10JIE = -0.9078631
# log10JII = -1.4561069
sig2 = 0.00095
s_n = 0.05 * np.sqrt(sig2)
s_b = 2.2990687 * np.sqrt(sig2)
broad_frac_e = 1.3001949
broad_frac_i = 1.3414183
log10JEE = -0.60524154
log10JEI = -1.4209754
log10JIE = -0.59155685
log10JII = -1.5301344

w_prm_dict = {
    's_n':          s_n,
    's_b':          s_b,
    'broad_frac_e': broad_frac_e,
    'broad_frac_i': broad_frac_i,
}

w_prm_dict.update({
    'wff_sum': wff_sum,
    'inh_inp_fact': 0.0,
    'wee_sum': 10**log10JEE * (1+w_prm_dict['broad_frac_e']),
    'wei_sum': 10**log10JEI * (1+w_prm_dict['broad_frac_i']),
    'wie_sum': 10**log10JIE * (1+w_prm_dict['broad_frac_e']),
    'wii_sum': 10**log10JII * (1+w_prm_dict['broad_frac_i']),
})

def init_net(
    n_iter: int,
    ):
    # compute number of LGN cells
    lgn_file = lgn_dir + 'seed={:d}.pkl'.format((n_iter + seed) % max_spike_file)
    print('Opening spike counts from',lgn_file)
    with open(lgn_file, 'rb') as handle:
        lgn_dict = pickle.load(handle)
    lgn_spikes = lgn_dict['spikes']
    n_lgn = lgn_spikes.shape[-1]
    n_x = n_lgn // n_grid**2 // 2
    n_stim = lgn_spikes.shape[0]
    print(n_lgn,'LGN cells')

    if n_iter==0: # starting a new simulation, must initialize the system
        net = Model(n_grid=n_grid,n_e=n_e,n_i=n_i,n_x=n_x,seed=seed,
                    s_x=s_x,s_s=s_s,gain_i=gain_i,
                    prune=prune,rec_e_plast=False,rec_i_plast=rec_plast,rec_i_ltd=rec_i_ltd,
                    w_prm_dict=w_prm_dict,
                    rx_wave_start=lgn_spikes[13])#lgn_spikes[26])
    else:
        # load weights, inputs, rates, averages, and learning rates from previous iteration
        with open(res_dir + 'seed={:d}_iter={:d}.pkl'.format(seed,n_iter-1), 'rb') as handle:
            res_dict = pickle.load(handle)
            
        net = Model(n_grid=n_grid,n_e=n_e,n_i=n_i,n_x=n_x,
                    s_x=s_x,s_s=s_s,gain_i=gain_i,
                    prune=prune,rec_e_plast=False,rec_i_plast=rec_plast,rec_i_ltd=rec_i_ltd,
                    w_prm_dict=w_prm_dict,init_dict=res_dict)
        
    return net
    
def run_iter(
    n_iter: int,
    net,
    save: bool=False
    ):
    print('\nRunning iteration',n_iter,'\n')
    
    # compute number of LGN cells
    lgn_file = lgn_dir + 'seed={:d}.pkl'.format((n_iter + seed) % max_spike_file)
    print('Opening spike counts from',lgn_file)
    with open(lgn_file, 'rb') as handle:
        lgn_dict = pickle.load(handle)
    lgn_spikes = lgn_dict['spikes']
    n_lgn = lgn_spikes.shape[-1]
    n_frame = lgn_spikes.shape[0]
    print(n_lgn,'LGN cells')

    # set inhibitory input level to slowly ramp with development
    net.inh_inp_fact = np.fmin(1,n_iter/50)

    start = time.process_time()
    for idx in range(n_frame):
        rx = lgn_spikes[idx]
        inh_mult = 1.0
        # if n_iter<10:
        #     inh_mult = 0.1 + 0.09*n_iter + 0.09*idx/n_frame
        # else:
        #     inh_mult = 1.0
        # if n_iter<10:
        #     inh_mult = 0.5 + 0.05*n_iter + 0.05*idx/n_frame
        # else:
        #     inh_mult = 1.0
        
        # update inputs and rates
        net.update_inps(rx,dt_stim,inh_mult)
        
        # update averages
        net.update_avgs(rx)
        
        # collect weight changes
        net.collect_dw(rx)
        
        if (idx+1)%n_batch==0:
            # update learning rates if during first two iterations
            if n_iter<50:
                net.update_learn_rates()
                # print(net.wex_rate,net.wee_rate,net.wei_rate)
                # print(net.wix_rate,net.wie_rate,net.wii_rate)
            
            if ((idx+1)//n_batch - 1)%5==0:
                print('dwex rms = {:.2e}, dwee rms = {:.2e}, dwei rms = {:.2e}'.format(
                    np.sqrt(np.mean(net.dwex**2)),np.sqrt(np.mean(net.dwee**2)),np.sqrt(np.mean(net.dwei**2))))
                print('dwix rms = {:.2e}, dwie rms = {:.2e}, dwii rms = {:.2e}'.format(
                    np.sqrt(np.mean(net.dwix**2)),np.sqrt(np.mean(net.dwie**2)),np.sqrt(np.mean(net.dwii**2))))
            
            # update weights
            net.sum_norm_dw()
            net.update_weights()
            
            # initialize weight change accumulators
            net.reset_dw()
            
            print('batch {:d} took'.format((idx+1)//n_batch),time.process_time() - start,'s')
            
            start = time.process_time()

    if save:
        res_file = res_dir + 'seed={:d}_iter={:d}.pkl'.format(seed,n_iter)
    
        res_dict = {}

        res_dict['wex'] = net.wex
        res_dict['wix'] = net.wix
        res_dict['wee'] = net.wee
        res_dict['wei'] = net.wei
        res_dict['wie'] = net.wie
        res_dict['wii'] = net.wii
        res_dict['wex_rate'] = net.wex_rate
        res_dict['wix_rate'] = net.wix_rate
        res_dict['wee_rate'] = net.wee_rate
        res_dict['wei_rate'] = net.wei_rate
        res_dict['wie_rate'] = net.wie_rate
        res_dict['wii_rate'] = net.wii_rate
        res_dict['uee'] = net.uee
        res_dict['uei'] = net.uei
        res_dict['uie'] = net.uie
        res_dict['uii'] = net.uii
        res_dict['uee_avg'] = net.uee_avg
        res_dict['uei_avg'] = net.uei_avg
        res_dict['uie_avg'] = net.uie_avg
        res_dict['uii_avg'] = net.uii_avg
        res_dict['rx_avg'] = net.rx_avg
        res_dict['thresh'] = net.thresh
        res_dict['max_prop_thresh'] = net.max_prop_thresh

        with open(res_file, 'wb') as handle:
            pickle.dump(res_dict,handle)
            
net = init_net(init_iter)

for n_iter in range(init_iter,init_iter+batch_iter):
    run_iter(n_iter,net,save=(n_iter+1)%batch_iter==0)

if init_iter+batch_iter < max_iter:
    os.system("python runjob_sim_2d_ssn_lgn_wave_rfs.py " + \
            "-ne {:d} -ni {:d} -iit {:d} -bit {:d} -mit {:d} -s {:d} -nw {:d} -ns {:d} -nh {:.2f} -ng {:d} -sx {:.2f} -ss {:.2f} -gi {:.1f} -w {:.2f} -p {:d} -r {:d} -d {:.1f} -m {:s}".format(
            n_e,n_i,init_iter+batch_iter,batch_iter,max_iter,
            seed,n_wave,n_stim,n_shrink,n_grid,
            s_x,s_s,gain_i,wff_sum,prune,rec_plast,rec_i_ltd,mode))
