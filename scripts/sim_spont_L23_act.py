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
from scipy import integrate
from scipy.stats import norm,gamma

parser = argparse.ArgumentParser()
parser.add_argument('--n_patt', '-npt', help='number of spontaneous patterns',type=int, default=50)
parser.add_argument('--n_int', '-nt', help='number of integration steps between phases',type=int, default=4)
parser.add_argument('--patt_cv', '-pcv', help='input coefficient of variation',type=float, default=0.65)
parser.add_argument('--spat_freq', '-sf', help='input spatial frequency decay length',type=int, default=8)
parser.add_argument('--inp_str', '-istr', help='mean of feedforward input',type=float, default=1.0)
parser.add_argument('--inp_seed', '-is', help='seed for L4 model input',type=int, default=0)
parser.add_argument('--rec_seed', '-rs', help='seed for recurrent connectivity',type=int, default=0)
parser.add_argument('--saverates', '-r', help='save rates or not',type=bool, default=False)
args = vars(parser.parse_args())
n_patt = int(args['n_patt'])
# n_rpt = int(args['n_rpt'])
n_int= int(args['n_int'])
patt_cv = args['patt_cv']
spat_freq = args['spat_freq']
inp_str = args['inp_str']
inp_seed = int(args['inp_seed'])
rec_seed = int(args['rec_seed'])
saverates = args['saverates']

N = 60

# Define parameters for connectivity
params = np.load("./../notebooks/l23_params.npy")

# Define where to save results
res_dir = './../results/'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

res_dir = res_dir + 'L23_sel/'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

res_file = res_dir + 'spont_cv={:.2f}_spat_freq={:d}_inp_str={:.1f}_inp_seed={:d}_rec_seed={:d}.pkl'.format(patt_cv,spat_freq,inp_str,inp_seed,rec_seed)

res_dict = {}

# Compute distance matrix for connectivity kernel
xs,ys = np.meshgrid(np.arange(N)/N,np.arange(N)/N)
dxs = np.abs(xs[:,:,None,None] - xs[None,None,:,:])
dxs[dxs > 0.5] = 1 - dxs[dxs > 0.5]
dys = np.abs(ys[:,:,None,None] - ys[None,None,:,:])
dys[dys > 0.5] = 1 - dys[dys > 0.5]
dss = np.sqrt(dxs**2 + dys**2).reshape(N**2,N**2)

nbins = 50

idxs = np.digitize(dss,np.linspace(0,np.max(dss),nbins+1))

npatt = n_patt
patts_fft = np.fft.fft2(np.random.default_rng(inp_seed).normal(size=(npatt,N,N)))
patts_fft[:,0,0] = 0 # remove DC component
freqs = np.fft.fftfreq(N,1/N)
freqs = np.sqrt(freqs[:,None]**2 + freqs[None,:]**2)

decay = spat_freq
patts_fft *= np.exp(-0.5*freqs**2/decay**2)[None,:,:]

patts = np.real(np.fft.ifft2(patts_fft).reshape(npatt,-1))
for i in range(10):
    patts -= np.mean(patts,axis=-1,keepdims=True)
    patts /= np.std(patts,axis=-1,keepdims=True)
    
    patts -= np.mean(patts,axis=0,keepdims=True)
    patts /= np.std(patts,axis=0,keepdims=True)
    
gam_dist = gamma(a=1/(patt_cv**2),scale=patt_cv**2)
patts = gam_dist.ppf(norm.cdf(patts))

freqs = np.fft.fftfreq(N,1/N)
freqs = np.sqrt(freqs[:,None]**2 + freqs[None,:]**2)
decay = 15
noise_filter = np.ones((N,N,N,N)) * np.exp(-0.5*freqs**2/decay**2)[:,:,None,None]

def gen_noise(rng):
    noise = rng.normal(size=(N,N,N,N))
    noise = np.fft.fftn(noise)
    noise *= noise_filter
    noise = np.real(np.fft.ifftn(noise))
    noise -= np.mean(noise)
    noise /= np.std(noise)
    return noise.reshape(N**2,N**2)

norm_dist = norm()

# define simulation functions
def integrate_sheet(xea0,xen0,xeg0,xia0,xin0,xig0,inp,Jee,Jei,Jie,Jii,kern_e,kern_i,
                    het_lev,N,ne,ni,threshe,threshi,
                    t0,dt,Nt,tsamp=None,ta=0.01,tn=0.300,tg=0.01,frac_n=0.7):
    '''
    Integrate 2D sheet with AMPA, NMDA, and GABA receptor dynamics.
    xe0, xi0: initial excitatory and inhibitory activity
    inp: input function, takes time t and returns input at that time
    Jee, Jei, Jie, Jii: connectivity strengths per connection type
    kern_e, kern_i: connectivity kernels for excitatory and inhibitory connections
    ne, ni: rate activation exponents for excitatory and inhibitory neurons
    threshe, threshi: activation thresholds for excitatory and inhibitory neurons
    t0: initial time
    dt: time step for integration
    Nt: number of time steps to integrate
    ta, tn, tg: time constants for AMPA, NMDA, and GABA receptor dynamics
    frac_n: fraction of NMDA vs NMDA+AMPA receptors in the excitatory population
    '''
    
    if tsamp is None:
        tsamp = [Nt-1]
    samp_idx = 0
    
    xea = xea0.copy()
    xen = xen0.copy()
    xeg = xeg0.copy()
    xia = xia0.copy()
    xin = xin0.copy()
    xig = xig0.copy()
    
    rng = np.random.default_rng(rec_seed)
    gam_dist = gamma(a=1/(het_lev**2),scale=het_lev**2)
    
    Wee = Jee*kern_e.reshape(N**2,N**2)*gam_dist.ppf(norm_dist.cdf(gen_noise(rng)))
    Wei = Jei*kern_i.reshape(N**2,N**2)*gam_dist.ppf(norm_dist.cdf(gen_noise(rng)))
    Wie = Jie*kern_e.reshape(N**2,N**2)*gam_dist.ppf(norm_dist.cdf(gen_noise(rng)))
    Wii = Jii*kern_i.reshape(N**2,N**2)*gam_dist.ppf(norm_dist.cdf(gen_noise(rng)))
    
    if len(xea.shape) == 1:
        xea = xea
        xen = xen
        xeg = xeg
        xia = xia
        xin = xin
        xig = xig
    
    resps = np.zeros((2,N**2,len(tsamp)))
        
    def dyn_func(t,x,ncell):
        xea = x[0*ncell:1*ncell]
        xen = x[1*ncell:2*ncell]
        xeg = x[2*ncell:3*ncell]
        xia = x[3*ncell:4*ncell]
        xin = x[4*ncell:5*ncell]
        xig = x[5*ncell:6*ncell]
        
        ff_inp = inp(t)

        ye = np.fmin(1e5,np.fmax(0,xea+xen+xeg-threshe)**ne)
        yi = np.fmin(1e5,np.fmax(0,xia+xin+xig-threshi)**ni)
        
        net_ee = Wee@ye + ff_inp
        net_ei = Wei@yi
        net_ie = Wie@ye + ff_inp
        net_ii = Wii@yi
        
        dx = np.zeros_like(x)
        dx[0*ncell:1*ncell] = ((1-frac_n)*net_ee - xea)/ta
        dx[1*ncell:2*ncell] = (frac_n*net_ee - xen)/tn
        dx[2*ncell:3*ncell] = (net_ei - xeg)/tg
        dx[3*ncell:4*ncell] = ((1-frac_n)*net_ie - xia)/ta
        dx[4*ncell:5*ncell] = (frac_n*net_ie - xin)/tn
        dx[5*ncell:6*ncell] = (net_ii - xig)/tg
        
        return dx.flatten()
    
    x0 = np.concatenate((xea,xen,xeg,xia,xin,xig),axis=0).flatten()
    
    start_time = time.process_time()
    max_time = 60
    def time_event(t,x,ncell):
        int_time = (start_time + max_time) - time.process_time()
        if int_time < 0: int_time = 0
        return int_time
    time_event.terminal = True
    
    sol = integrate.solve_ivp(dyn_func,(0,dt*Nt),y0=x0,t_eval=tsamp*dt,args=(N**2,),method='RK23')#,events=time_event)
    if sol.status != 0:
        x = np.nan*np.ones((6*N**2,len(tsamp)))
    else:
        x = sol.y
    x = x.reshape((-1,len(tsamp)))
    
    xea = x[0*N**2:1*N**2,:]
    xen = x[1*N**2:2*N**2,:]
    xeg = x[2*N**2:3*N**2,:]
    xia = x[3*N**2:4*N**2,:]
    xin = x[4*N**2:5*N**2,:]
    xig = x[5*N**2:6*N**2,:]
        
    ye = np.fmin(1e5,np.fmax(0,xea+xen+xeg-threshe)**ne)
    yi = np.fmin(1e5,np.fmax(0,xia+xin+xig-threshi)**ni)
    
    resps[0] = ye
    resps[1] = yi
    # return xea,xen,xeg,xia,xin,xig,np.concatenate((ye,yi))
    return resps

def get_sheet_resps(params,N):
    '''
    params[0] = det(J)/(|Jei| * |Jie|) = 1 - (|Jee| * |Jii|) / (|Jei| * |Jie|)
    params[1] = (|Jee|-|Jii|)/(|Jei| + |Jie|)
    params[2] = (log10[|Jei|] + log10[|Jie|]) / 2
    params[3] = (log10[|Jei|] - log10[|Jie|]) / 2
    params[4] = s_e
    params[5] = s_i
    params[6] = het_level
    params[7] = base_e
    params[8] = base_i
    '''
    Jee,Jei,Jie,Jii = 10**params[:4]
    Jei *= -1
    Jii *= -1
    
    nint = 5
    nwrm = 6 * nint
    dt = 1 / (nint * 3)
    
    kern_e = np.exp(-(dss/params[4])**2)
    norm = kern_e.sum(axis=1).mean(axis=0)
    kern_e /= norm
    
    kern_i = np.exp(-(dss/params[5])**2)
    norm = kern_i.sum(axis=1).mean(axis=0)
    kern_i /= norm
    
    thresh_e = -params[7]
    thresh_i = -params[8]
    
    tsamp = np.array([nwrm-1])
    resps = np.zeros((2,N**2,npatt))
    for patt_idx,patt in tqdm(enumerate(patts)):
        def ff_inp(t):
            return inp_str*patt
        resps[:,:,patt_idx] = integrate_sheet(np.zeros(N**2),np.zeros(N**2),np.zeros(N**2),
                                np.zeros(N**2),np.zeros(N**2),np.zeros(N**2),
                                ff_inp,Jee,Jei,Jie,Jii,
                                kern_e,kern_i,params[6],N,2,2,
                                thresh_e,thresh_i,0,dt,nwrm,tsamp)[:,:,-1]
    return resps

# Integrate to get firing rates
start = time.process_time()

L23_rates = get_sheet_resps(params,N)
    
print('Simulating rate dynamics took',time.process_time() - start,'s\n')

if saverates:
    res_dict['L23_rates'] = L23_rates

# Calculate CV of inputs and responses
L23_rate_r0 = np.mean(L23_rates,(-2,-1))

res_dict['L23_rate_r0'] = L23_rate_r0

with open(res_file, 'wb') as handle:
    pickle.dump(res_dict,handle)
