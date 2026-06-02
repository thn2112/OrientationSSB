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
from scipy.interpolate import interp1d
from scipy import integrate

import analyze_func as af
import map_func as mf

parser = argparse.ArgumentParser()
parser.add_argument('--n_ori', '-no', help='number of orientations',type=int, default=16)
parser.add_argument('--n_phs', '-np', help='number of orientations',type=int, default=16)
parser.add_argument('--n_int', '-nt', help='number of integration steps between phases',type=int, default=4)
parser.add_argument('--map', '-m', help='type of map',type=str, default=None)
parser.add_argument('--static', '-st', help='static or dynamic input',type=bool, default=False)
parser.add_argument('--num_seeds', '-s', help='number of seeds to average over',type=int, default=0)
parser.add_argument('--num_samps', '-sa', help='number of samples from each seed to save',type=int, default=100)
args = vars(parser.parse_args())
n_ori = int(args['n_ori'])
n_phs = int(args['n_phs'])
# n_rpt = int(args['n_rpt'])
n_int= int(args['n_int'])
static = args['static']
num_seeds = int(args['num_seeds'])
num_samps = int(args['num_samps'])

N = 60

# Define where to save results
res_dir = './../results/'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

res_dir = res_dir + 'L4_sel/'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

if static:
    res_dir = res_dir + 'static_'

if args['map'] is not None:
    res_dir = res_dir + '{:s}_'.format(args['map'])

res_dict = {}
rng = np.random.default_rng(0)

nbins = 43

samp_idxs = np.ones((num_seeds,num_samps),dtype=int)
inp_os_samps = np.ones((num_seeds,num_samps))*np.nan
inp_po_samps = np.ones((num_seeds,num_samps))*np.nan
inp_mr_samps = np.ones((num_seeds,num_samps))*np.nan
inp_fpss = np.ones((num_seeds,nbins))*np.nan
inp_po_fpss = np.ones((num_seeds,nbins))*np.nan
rate_os_samps = np.ones((num_seeds,num_samps))*np.nan
rate_po_samps = np.ones((num_seeds,num_samps))*np.nan
rate_mr_samps = np.ones((num_seeds,num_samps))*np.nan
rate_fpss = np.ones((num_seeds,nbins))*np.nan
rate_po_fpss = np.ones((num_seeds,nbins))*np.nan
mismatch_samps = np.ones((num_seeds,num_samps))*np.nan
ff_rec_ori_mm_samps = np.ones((num_seeds,num_samps))*np.nan
ff_rec_phs_mm_samps = np.ones((num_seeds,num_samps,n_ori))*np.nan
corr_curves = np.ones((num_seeds,nbins))*np.nan
mods = np.ones(num_seeds)*np.nan
dims = np.ones(num_seeds)*np.nan

for seed_idx in range(num_seeds):
    samp_idxs[seed_idx] = rng.choice(N**2,size=num_samps,replace=False)
    
    try:
        with open(res_dir + 'seed={:d}.pkl'.format(seed_idx),'rb') as handle:
            file_dict = pickle.load(handle)
    except:
        continue
        
    inp_opm = file_dict['inp_opm'][0].reshape(N,N)
    inp_mr = file_dict['inp_mr'][0].reshape(N,N)
    rate_opm = file_dict['L4_rate_opm'][0].reshape(N,N)
    rate_mr = file_dict['L4_rate_mr'][0].reshape(N,N)
    ff_opm = file_dict['ff_opm']
    ff_ppms = file_dict['ff_ppms']
    rec_opm = file_dict['rec_opm']
    rec_ppms = file_dict['rec_ppms']
    net_opm = file_dict['net_opm']
    net_ppms = file_dict['net_ppms']

    inp_po = np.angle(inp_opm)*180/(2*np.pi)
    inp_po[inp_po > 90] -= 180
    rate_po = np.angle(rate_opm)*180/(2*np.pi)
    rate_po[rate_po > 90] -= 180
    ff_po = np.angle(ff_opm)*180/(2*np.pi)
    ff_po[ff_po > 90] -= 180
    rec_po = np.angle(rec_opm)*180/(2*np.pi)
    rec_po[rec_po > 90] -= 180
    ff_pps = np.angle(ff_ppms)*360/(2*np.pi)
    rec_pps = np.angle(rec_ppms)*360/(2*np.pi)

    inp_os_samp = np.abs(inp_opm).flatten()[samp_idxs[seed_idx]]
    rate_os_samp = np.abs(rate_opm).flatten()[samp_idxs[seed_idx]]
    inp_po_samp = inp_po.flatten()[samp_idxs[seed_idx]]
    rate_po_samp = rate_po.flatten()[samp_idxs[seed_idx]]
    inp_mr_samp = inp_mr.flatten()[samp_idxs[seed_idx]]
    rate_mr_samp = rate_mr.flatten()[samp_idxs[seed_idx]]
    mismatch_samp = np.abs(inp_po_samp - rate_po_samp)
    mismatch_samp[mismatch_samp > 90] = 180 - mismatch_samp[mismatch_samp > 90]
    ff_rec_ori_mm_samp = np.abs(ff_po.flatten()[samp_idxs[seed_idx]] - rec_po.flatten()[samp_idxs[seed_idx]])
    ff_rec_ori_mm_samp[ff_rec_ori_mm_samp > 90] = 180 - ff_rec_ori_mm_samp[ff_rec_ori_mm_samp > 90]
    ff_rec_phs_mm_samp = np.abs(ff_pps.reshape(N**2,-1)[samp_idxs[seed_idx]] - rec_pps.reshape(N**2,-1)[samp_idxs[seed_idx]])
    ff_rec_phs_mm_samp[ff_rec_phs_mm_samp > 180] = 360 - ff_rec_phs_mm_samp[ff_rec_phs_mm_samp > 180]

    _,inp_fps = af.get_fps(inp_opm,nbins=nbins)
    _,rate_fps = af.get_fps(rate_opm,nbins=nbins)
    _,inp_po_fps = af.get_fps(inp_opm/np.abs(inp_opm),nbins=nbins)
    _,rate_po_fps = af.get_fps(rate_opm/np.abs(rate_opm),nbins=nbins)
    corr,corr_curve = af.get_corr(file_dict['L4_rates'][0].reshape(N**2,-1),nbins=nbins)
    arg_min = np.argmin(corr_curve)
    corr_mins = corr_curve[arg_min]
    corr_maxs = np.max(corr_curve[arg_min:])
    mod = corr_maxs - corr_mins
    
    dim = np.trace(corr)**2 / np.trace(corr @ corr)
    
    inp_os_samps[seed_idx] = inp_os_samp
    inp_po_samps[seed_idx] = inp_po_samp
    inp_mr_samps[seed_idx] = inp_mr_samp
    inp_fpss[seed_idx] = inp_fps
    inp_po_fpss[seed_idx] = inp_po_fps
    rate_os_samps[seed_idx] = rate_os_samp
    rate_po_samps[seed_idx] = rate_po_samp
    rate_mr_samps[seed_idx] = rate_mr_samp
    rate_fpss[seed_idx] = rate_fps
    rate_po_fpss[seed_idx] = rate_po_fps
    mismatch_samps[seed_idx] = mismatch_samp
    ff_rec_ori_mm_samps[seed_idx] = ff_rec_ori_mm_samp
    ff_rec_phs_mm_samps[seed_idx] = ff_rec_phs_mm_samp
    corr_curves[seed_idx] = corr_curve
    mods[seed_idx] = mod
    dims[seed_idx] = dim
    
res_dict['samp_idxs'] = samp_idxs
res_dict['inp_os_samps'] = inp_os_samps
res_dict['inp_po_samps'] = inp_po_samps
res_dict['inp_mr_samps'] = inp_mr_samps
res_dict['inp_fpss'] = inp_fpss
res_dict['inp_po_fpss'] = inp_po_fpss
res_dict['rate_os_samps'] = rate_os_samps
res_dict['rate_po_samps'] = rate_po_samps
res_dict['rate_mr_samps'] = rate_mr_samps
res_dict['rate_fpss'] = rate_fpss
res_dict['rate_po_fpss'] = rate_po_fpss
res_dict['mismatch_samps'] = mismatch_samps
res_dict['ff_rec_ori_mm_samps'] = ff_rec_ori_mm_samps
res_dict['ff_rec_phs_mm_samps'] = ff_rec_phs_mm_samps
res_dict['corr_curves'] = corr_curves
res_dict['mods'] = mods
res_dict['dims'] = dims

with open(res_dir + 'analysis.pkl', 'wb') as handle:
    pickle.dump(res_dict,handle)
