import sys
import os
try:
    import pickle5 as pickle
except:
    import pickle
sys.path.insert(0, './..')

import argparse
import time
from tqdm import tqdm

import numpy as np
from scipy.stats import norm,gamma

import analyze_func as af

parser = argparse.ArgumentParser()
parser.add_argument('--n_patt', '-npt', help='number of spontaneous patterns',type=int, default=50)
parser.add_argument('--n_int', '-nt', help='number of integration steps between phases',type=int, default=4)
parser.add_argument('--patt_cv', '-pcv', help='input coefficient of variation',type=float, default=0.65)
parser.add_argument('--num_seeds', '-s', help='number of seeds to average over',type=int, default=0)
args = vars(parser.parse_args())
n_patt = int(args['n_patt'])
patt_cv = float(args['patt_cv'])
# n_rpt = int(args['n_rpt'])
n_int= int(args['n_int'])
num_seeds = int(args['num_seeds'])

N = 60

# Define where to save results
res_dir = './../results/'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

res_dir = res_dir + 'L23_sel/'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

res_dict = {}
rng = np.random.default_rng(0)

nbins = 43
npatt = n_patt
xs,ys = np.meshgrid(np.arange(N)/N,np.arange(N)/N)
dxs = np.abs(xs[:,:,None,None] - xs[None,None,:,:])
dxs[dxs > 0.5] = 1 - dxs[dxs > 0.5]
dys = np.abs(ys[:,:,None,None] - ys[None,None,:,:])
dys[dys > 0.5] = 1 - dys[dys > 0.5]
dss = np.sqrt(dxs**2 + dys**2).reshape(N**2,N**2)

inp_corr_curves = np.ones((num_seeds,nbins))*np.nan
rate_corr_curves = np.ones((num_seeds,nbins))*np.nan
mods = np.ones(num_seeds)*np.nan
dims = np.ones(num_seeds)*np.nan

for seed_idx in tqdm(range(num_seeds)):
    try:
        with open(res_dir + 'spont_cv={:.2f}_seed={:d}.pkl'.format(patt_cv,seed_idx),'rb') as handle:
            file_dict = pickle.load(handle)
    except:
        continue
    
    patts_fft = np.fft.fft2(np.random.default_rng(seed_idx).normal(size=(npatt,N,N)))
    patts_fft[:,0,0] = 0 # remove DC component
    freqs = np.fft.fftfreq(N,1/N)
    freqs = np.sqrt(freqs[:,None]**2 + freqs[None,:]**2)

    decay = 8
    patts_fft *= np.exp(-0.5*freqs**2/decay**2)[None,:,:]

    patts = np.real(np.fft.ifft2(patts_fft).reshape(npatt,-1))
    for i in range(10):
        patts -= np.mean(patts,axis=-1,keepdims=True)
        patts /= np.std(patts,axis=-1,keepdims=True)
        
        patts -= np.mean(patts,axis=0,keepdims=True)
        patts /= np.std(patts,axis=0,keepdims=True)
        
    gam_dist = gamma(a=1/(patt_cv**2),scale=patt_cv**2)
    inps = gam_dist.ppf(norm.cdf(patts)).T.reshape(N**2,-1)
    
    _,inp_corr_curve = af.get_corr(inps,nbins=nbins)
    rate_corr,rate_corr_curve = af.get_corr(file_dict['L23_rates'][0].reshape(N**2,-1),nbins=nbins)
    
    arg_min = np.argmin(rate_corr_curve)
    corr_mins = rate_corr_curve[arg_min]
    corr_maxs = np.max(rate_corr_curve[arg_min:])
    mod = corr_maxs - corr_mins
    
    dim = np.trace(rate_corr)**2 / np.trace(rate_corr @ rate_corr)
    
    inp_corr_curves[seed_idx] = inp_corr_curve
    rate_corr_curves[seed_idx] = rate_corr_curve
    mods[seed_idx] = mod
    dims[seed_idx] = dim
    
res_dict['inp_corr_curves'] = inp_corr_curves
res_dict['rate_corr_curves'] = rate_corr_curves
res_dict['mods'] = mods
res_dict['dims'] = dims

with open(res_dir + f'spont_cv={patt_cv:.2f}_analysis.pkl', 'wb') as handle:
    pickle.dump(res_dict,handle)
