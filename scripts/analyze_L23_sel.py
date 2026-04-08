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
parser.add_argument('--add_phase', '-ap', help='add phase to L4 inputs or not',type=bool, default=False)
parser.add_argument('--add_orisel', '-aos', help='add orientation selectivity to L4 inputs or not',type=bool, default=False)
parser.add_argument('--add_sandp', '-asp', help='make L4 inputs salt and pepper or not',type=bool, default=False)
parser.add_argument('--num_seeds', '-s', help='number of seeds to average over',type=int, default=0)
parser.add_argument('--num_samps', '-sa', help='number of samples from each seed to save',type=int, default=100)
args = vars(parser.parse_args())
n_ori = int(args['n_ori'])
n_phs = int(args['n_phs'])
# n_rpt = int(args['n_rpt'])
n_int= int(args['n_int'])
static = args['static']
add_phase = args['add_phase']
add_orisel = args['add_orisel']
add_sandp = args['add_sandp']
num_seeds = int(args['num_seeds'])
num_samps = int(args['num_samps'])

N = 60

# Define where to save results
res_dir = './../results/'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

l4_dir = res_dir + 'L4_sel/'
res_dir = res_dir + 'L23_sel/'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

if static:
    res_dir = res_dir + 'static_'
    l4_dir = l4_dir + 'static_'

if args['map'] is not None:
    res_dir = res_dir + '{:s}_'.format(args['map'])
    L4_dir = l4_dir + '{:s}_'.format(args['map'])
    
if add_phase:
    res_dir = res_dir + 'phase_'
    
if add_orisel:
    res_dir = res_dir + 'orisel_'
    
if add_sandp:
    res_dir = res_dir + 'sandp_'

res_dict = {}
rng = np.random.default_rng(0)

nbins = 43
npatt = n_ori * n_phs
xs,ys = np.meshgrid(np.arange(N)/N,np.arange(N)/N)
dxs = np.abs(xs[:,:,None,None] - xs[None,None,:,:])
dxs[dxs > 0.5] = 1 - dxs[dxs > 0.5]
dys = np.abs(ys[:,:,None,None] - ys[None,None,:,:])
dys[dys > 0.5] = 1 - dys[dys > 0.5]
dss = np.sqrt(dxs**2 + dys**2).reshape(N**2,N**2)

idxs = np.digitize(dss,np.linspace(0,np.max(dss),nbins+1))
inp_os_samps = np.ones((num_seeds,num_samps))*np.nan
inp_po_samps = np.ones((num_seeds,num_samps))*np.nan
inp_mr_samps = np.ones((num_seeds,num_samps))*np.nan
inp_fpss = np.ones((num_seeds,nbins))*np.nan
inp_corr_curves = np.ones((num_seeds,nbins))*np.nan
rate_os_samps = np.ones((num_seeds,num_samps))*np.nan
rate_po_samps = np.ones((num_seeds,num_samps))*np.nan
rate_mr_samps = np.ones((num_seeds,num_samps))*np.nan
rate_fpss = np.ones((num_seeds,nbins))*np.nan
rate_corr_curves = np.ones((num_seeds,nbins))*np.nan
mismatch_samps = np.ones((num_seeds,num_samps))*np.nan
mods = np.ones(num_seeds)*np.nan
dims = np.ones(num_seeds)*np.nan

with open(l4_dir + 'analysis.pkl', 'rb') as handle:
    l4_res_dict = pickle.load(handle)
    
samp_idxs = l4_res_dict['samp_idxs']

for seed_idx in range(num_seeds):
    try:
        with open(res_dir + 'seed={:d}.pkl'.format(seed_idx),'rb') as handle:
            file_dict = pickle.load(handle)
    except:
        continue
    
    try:
        with open(l4_dir + 'seed={:d}.pkl'.format(seed_idx),'rb') as handle:
            l4_dict = pickle.load(handle)
    except:
        continue
    
    l4_rates = l4_dict['L4_rates']
    l4_rates /= np.nanmean(l4_rates,axis=(-2,-1),keepdims=True)
    if add_phase:
        _,_,phs = af.calc_dc_ac_comp(l4_rates)
        l4_phase_rates = np.fmax(0,np.cos(np.linspace(0,2*np.pi,8,endpoint=False)[None,None,None,:]-phs[:,:,:,None]))
        l4_phase_rates *= np.nanmean(l4_rates,axis=(-1),keepdims=True) \
            / np.nanmean(l4_phase_rates,axis=(-1),keepdims=True)
        l4_rates = l4_phase_rates
    if add_orisel:
        _,_,doub_po = af.calc_dc_ac_comp(l4_rates.mean(-1))
        l4_orisel_rates = np.fmax(0,np.cos(np.linspace(0,2*np.pi,8,endpoint=False)[None,None,:]-doub_po[:,:,None]))
        l4_orisel_rates *= np.nanmean(l4_rates.mean(-1),axis=(-1),keepdims=True) / np.nanmean(l4_orisel_rates,axis=(-1),keepdims=True)
        l4_norm_phase_tuning = np.fmax(1e-12,l4_rates / np.nanmean(l4_rates,axis=(-1),keepdims=True))
        l4_rates = l4_norm_phase_tuning * l4_orisel_rates[:,:,:,None]
    if add_sandp:
        rng = np.random.default_rng(seed_idx)
        l4_rates[0] = rng.permutation(l4_rates[0])
        
    l4_rate_opm,l4_rate_MR = af.calc_OPM_MR(l4_rates)
    l4_dict['L4_rate_opm'] = l4_rate_opm.reshape(2,-1)
    l4_dict['L4_rate_mr'] = l4_rate_MR.reshape(2,-1)
    l4_dict['L4_rates'] = l4_rates * np.nanmean(l4_dict['L4_rates'],axis=(-2,-1),keepdims=True)
    
    inp_opm = l4_dict['L4_rate_opm'][0].reshape(N,N)
    inp_mr = l4_dict['L4_rate_mr'][0].reshape(N,N)
    rate_opm = file_dict['L23_rate_opm'][0].reshape(N,N)
    rate_mr = file_dict['L23_rate_mr'][0].reshape(N,N)

    inp_po = np.angle(inp_opm)*180/(2*np.pi)
    inp_po[inp_po > 90] -= 180
    rate_po = np.angle(rate_opm)*180/(2*np.pi)
    rate_po[rate_po > 90] -= 180
    
    inp_os_samp = np.abs(inp_opm).flatten()[samp_idxs[seed_idx]]
    rate_os_samp = np.abs(rate_opm).flatten()[samp_idxs[seed_idx]]
    inp_po_samp = inp_po.flatten()[samp_idxs[seed_idx]]
    rate_po_samp = rate_po.flatten()[samp_idxs[seed_idx]]
    inp_mr_samp = inp_mr.flatten()[samp_idxs[seed_idx]]
    rate_mr_samp = rate_mr.flatten()[samp_idxs[seed_idx]]
    mismatch_samp = np.abs(inp_po_samp - rate_po_samp)
    mismatch_samp[mismatch_samp > 90] = 180 - mismatch_samp[mismatch_samp > 90]

    _,inp_fps = af.get_fps(inp_opm)
    _,rate_fps = af.get_fps(rate_opm)
    
    _,inp_corr_curve = af.get_corr(l4_rates[0].reshape(N**2,-1),nbins=nbins)
    rate_corr,rate_corr_curve = af.get_corr(file_dict['L23_rates'][0].reshape(N**2,-1),nbins=nbins)
    
    arg_min = np.argmin(rate_corr_curve)
    corr_mins = rate_corr_curve[arg_min]
    corr_maxs = np.max(rate_corr_curve[arg_min:])
    mod = corr_maxs - corr_mins
    
    dim = np.trace(rate_corr)**2 / np.trace(rate_corr @ rate_corr)
    
    inp_os_samps[seed_idx] = inp_os_samp
    inp_po_samps[seed_idx] = inp_po_samp
    inp_mr_samps[seed_idx] = inp_mr_samp
    inp_fpss[seed_idx] = inp_fps
    inp_corr_curves[seed_idx] = inp_corr_curve
    rate_os_samps[seed_idx] = rate_os_samp
    rate_po_samps[seed_idx] = rate_po_samp
    rate_mr_samps[seed_idx] = rate_mr_samp
    rate_fpss[seed_idx] = rate_fps
    rate_corr_curves[seed_idx] = rate_corr_curve
    mismatch_samps[seed_idx] = mismatch_samp
    mods[seed_idx] = mod
    dims[seed_idx] = dim
    
res_dict['inp_os_samps'] = inp_os_samps
res_dict['inp_po_samps'] = inp_po_samps
res_dict['inp_mr_samps'] = inp_mr_samps
res_dict['inp_fpss'] = inp_fpss
res_dict['inp_corr_curves'] = inp_corr_curves
res_dict['rate_os_samps'] = rate_os_samps
res_dict['rate_po_samps'] = rate_po_samps
res_dict['rate_mr_samps'] = rate_mr_samps
res_dict['rate_fpss'] = rate_fpss
res_dict['rate_corr_curves'] = rate_corr_curves
res_dict['mismatch_samps'] = mismatch_samps
res_dict['mods'] = mods
res_dict['dims'] = dims

with open(res_dir + 'analysis.pkl', 'wb') as handle:
    pickle.dump(res_dict,handle)
