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

if args['map'] is None:
    res_file = res_dir + 'analysis.pkl'
    file_dir = res_dir
else:
    res_file = res_dir + '{:s}_analysis.pkl'.format(args['map'])
    file_dir = res_dir + '{:s}_'.format(args['map'])

res_dict = {}
rng = np.random.default_rng(0)

nbins = 50
npatt = n_ori * n_phs
xs,ys = np.meshgrid(np.arange(N)/N,np.arange(N)/N)
dxs = np.abs(xs[:,:,None,None] - xs[None,None,:,:])
dxs[dxs > 0.5] = 1 - dxs[dxs > 0.5]
dys = np.abs(ys[:,:,None,None] - ys[None,None,:,:])
dys[dys > 0.5] = 1 - dys[dys > 0.5]
dss = np.sqrt(dxs**2 + dys**2).reshape(N**2,N**2)

idxs = np.digitize(dss,np.linspace(0,np.max(dss),nbins+1))

inp_os_samps = np.ones((num_seeds,num_samps))*np.nan
rate_os_samps = np.ones((num_seeds,num_samps))*np.nan
inp_mr_samps = np.ones((num_seeds,num_samps))*np.nan
rate_mr_samps = np.ones((num_seeds,num_samps))*np.nan
mismatch_samps = np.ones((num_seeds,num_samps))*np.nan
rate_fpss = np.ones((num_seeds,N//2))*np.nan
inp_fpss = np.ones((num_seeds,N//2))*np.nan
corr_curves = np.ones((num_seeds,nbins))*np.nan
mods = np.ones(num_seeds)*np.nan
dims = np.ones(num_seeds)*np.nan

for seed_idx in range(num_seeds):
    samp_idxs = rng.choice(N**2,size=num_samps,replace=False)
    
    try:
        with open(file_dir + 'seed={:d}.pkl'.format(seed_idx),'rb') as handle:
            file_dict = pickle.load(handle)
    except:
        continue
        
    inp_opm = file_dict['inp_opm'][0].reshape(N,N)
    inp_mr = file_dict['inp_mr'][0].reshape(N,N)
    rate_opm = file_dict['L4_rate_opm'][0].reshape(N,N)
    rate_mr = file_dict['L4_rate_mr'][0].reshape(N,N)

    inp_PO = np.angle(inp_opm)*180/(2*np.pi)
    inp_PO[inp_PO > 90] -= 180
    rate_PO = np.angle(rate_opm)*180/(2*np.pi)
    rate_PO[rate_PO > 90] -= 180

    mismatch = np.abs(inp_PO - rate_PO)
    mismatch[mismatch > 90] = 180 - mismatch[mismatch > 90]
    
    inp_os_samp = np.abs(inp_opm).flatten()[samp_idxs]
    rate_os_samp = np.abs(rate_opm).flatten()[samp_idxs]
    inp_mr_samp = inp_mr.flatten()[samp_idxs]
    rate_mr_samp = rate_mr.flatten()[samp_idxs]
    mismatch_samp = mismatch.flatten()[samp_idxs]

    rate_fft = np.abs(np.fft.fftshift(np.fft.fft2(rate_opm - np.nanmean(rate_opm))))**2
    rate_fps = np.zeros(N//2)
    inp_fft = np.abs(np.fft.fftshift(np.fft.fft2(inp_opm - np.nanmean(inp_opm))))**2
    inp_fps = np.zeros(N//2)

    grid = np.arange(-N//2,N//2)
    x,y = np.meshgrid(grid,grid)
    bin_idxs = np.digitize(np.sqrt(x**2+y**2),np.arange(0,np.ceil(N//2*np.sqrt(2)))+0.5)
    for idx in range(N//2):
        rate_fps[idx] = np.mean(rate_fft[bin_idxs == idx])
        inp_fps[idx] = np.mean(inp_fft[bin_idxs == idx])
    
    resp_z = file_dict['L4_rates'][0].reshape(N**2,-1)
    resp_z = resp_z - np.mean(resp_z,axis=-1,keepdims=True)
    resp_z = resp_z / np.std(resp_z,axis=-1,keepdims=True)
    corr = np.zeros((N**2,N**2))
    for i in range(npatt):
        corr += resp_z[None,:,i] * resp_z[:,None,i]
    corr /= npatt
    
    corr_curve = np.zeros((nbins,))
    for i in range(nbins):
        corr_curve[i] = np.mean(corr[idxs == i+1],axis=-1)
    arg_min = np.argmin(corr_curve)
    corr_mins = corr_curve[arg_min]
    corr_maxs = np.max(corr_curve[arg_min:])
    mod = corr_maxs - corr_mins
    
    dim = np.trace(corr)**2 / np.trace(corr @ corr)
    
    inp_os_samps[seed_idx] = inp_os_samp
    rate_os_samps[seed_idx] = rate_os_samp
    inp_mr_samps[seed_idx] = inp_mr_samp
    rate_mr_samps[seed_idx] = rate_mr_samp
    mismatch_samps[seed_idx] = mismatch_samp
    rate_fpss[seed_idx] = rate_fps
    inp_fpss[seed_idx] = inp_fps
    corr_curves[seed_idx] = corr_curve
    mods[seed_idx] = mod
    dims[seed_idx] = dim
    
res_dict['inp_os_samps'] = inp_os_samps
res_dict['rate_os_samps'] = rate_os_samps
res_dict['inp_mr_samps'] = inp_mr_samps
res_dict['rate_mr_samps'] = rate_mr_samps
res_dict['mismatch_samps'] = mismatch_samps
res_dict['rate_fpss'] = rate_fpss
res_dict['inp_fpss'] = inp_fpss
res_dict['corr_curves'] = corr_curves
res_dict['mods'] = mods
res_dict['dims'] = dims

with open(res_file, 'wb') as handle:
    pickle.dump(res_dict,handle)
