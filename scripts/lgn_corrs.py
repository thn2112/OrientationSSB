import os
import pickle
import time
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--n_wave', '-nw', help='number of geniculate waves',type=int, default=15)
parser.add_argument('--n_stim', '-ns', help='number of light/dark sweeping bars',type=int, default=2)
parser.add_argument('--n_shrink', '-nh', help='factor by which to shrink stimuli',type=float, default=1.0)
parser.add_argument('--n_grid', '-ng', help='number of points per grid edge',type=int, default=20)
args = vars(parser.parse_args())
ngrid = int(args['n_grid'])
nwave = int(args['n_wave'])
nstim = int(args['n_stim'])
nshrink = float(args['n_shrink'])
nwave = int(args['n_wave'])

res_file = './../results/lgn_corrs_nw={:d}_ns={:d}_nh={:.2f}_ng={:d}.pkl'.format(nwave,nstim,nshrink,ngrid)

xs,ys = np.meshgrid(np.linspace(0.5/ngrid,1-0.5/ngrid,ngrid),
                    np.linspace(0.5/ngrid,1-0.5/ngrid,ngrid))
xs,ys = xs.flatten(),ys.flatten()
dists = np.sqrt(np.fmin(np.abs(xs[:,None]-xs[None,:]),
                        1-np.abs(xs[:,None]-xs[None,:]))**2 +\
                np.fmin(np.abs(ys[:,None]-ys[None,:]),
                        1-np.abs(ys[:,None]-ys[None,:]))**2)

max_val = np.round(ngrid/np.sqrt(2)).astype(int)
nbins = np.round(ngrid/np.sqrt(2)).astype(int)
step = max_val/ngrid / nbins
dist_bins = np.digitize(dists,np.linspace(0,max_val/ngrid,nbins+1) + step/2)

def comp_corrs(mode,time_mask=None):
    spikes = np.zeros((1,2*ngrid**2),np.ushort)
    for i in range(50):
        with open('./../results/2d_lgn_{:s}_spikes_nw={:d}_ns={:d}_nh={:.2f}_ng={:d}/seed={:d}.pkl'.\
            format(mode,nwave,nstim,nshrink,ngrid,i), 'rb') as handle:
            res_dict = pickle.load(handle)
        spikes = np.concatenate((spikes,res_dict['spikes']),axis=0)
        
    dt = 0.1

    times = np.arange(len(spikes)) * 0.1
    sweeping = (np.mod(np.round(times/dt),np.round(3.6/dt)) >= np.round(1.2/dt)) \
        & (np.mod(np.round(times/dt),np.round(3.6/dt)) <= 2*np.round(1.2/dt))
        
    if time_mask == 'sweeping':
        spikes = spikes[:,sweeping]
    elif time_mask == 'non_sweeping':
        spikes = spikes[:,~sweeping]
    corrs = np.corrcoef(spikes.T)
    
    corr_curve = np.zeros((2,nbins))
    
    for i in range(nbins):
        idxs = dist_bins == i
        corr_curve[0,i] += 0.5*np.mean(corrs[:ngrid**2,:ngrid**2][idxs])
        corr_curve[0,i] += 0.5*np.mean(corrs[ngrid**2:,ngrid**2:][idxs])

        corr_curve[1,i] += 0.5*np.mean(corrs[:ngrid**2,ngrid**2:][idxs])
        corr_curve[1,i] += 0.5*np.mean(corrs[ngrid**2:,:ngrid**2][idxs])
        
    return corr_curve

res_dict = {}
for mode,time_mask in zip(['spont','vis','spont_vis'],['non_sweeping','sweeping',None]):
    try:
        corr_curve = comp_corrs(mode)
        res_dict[mode] = corr_curve
    except:
        print('Error in mode {:s}'.format(mode))
    
with open(res_file,'wb') as handle:
    pickle.dump(res_dict, handle)