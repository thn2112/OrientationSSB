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
from scipy.interpolate import CubicSpline,PchipInterpolator
from scipy import integrate
from scipy.signal import argrelmin,argrelmax
from scipy.stats import norm,gamma

from tqdm import tqdm

import analyze_func as af

parser = argparse.ArgumentParser()
parser.add_argument('--n_ori', '-no', help='number of orientations',type=int, default=16)
parser.add_argument('--n_phs', '-np', help='number of orientations',type=int, default=16)
parser.add_argument('--n_int', '-nt', help='number of integration steps between phases',type=int, default=4)
parser.add_argument('--inp_seed', '-is', help='seed for L4 model input',type=int, default=0)
parser.add_argument('--rec_seed', '-rs', help='seed for recurrent connectivity',type=int, default=0)
parser.add_argument('--num_noise_seeds', '-ns', help='number of seeds for input noise',type=int, default=5)
parser.add_argument('--add_phase', '-ap', help='add phase selectivity to L4 inputs or not',type=bool, default=False)
parser.add_argument('--remove_phase', '-rp', help='remove phase selectivity from L4 inputs or not',type=bool, default=False)
parser.add_argument('--add_orisel', '-aos', help='add orientation selectivity to L4 inputs or not',type=bool, default=False)
parser.add_argument('--add_sandp', '-asp', help='make L4 inputs salt and pepper or not',type=bool, default=False)
parser.add_argument('--add_ffl4', '-aff', help='make L4 a FF model or not',type=bool, default=False)
parser.add_argument('--map', '-m', help='whether to switch to a different L4 map',type=str, default=None)
parser.add_argument('--static', '-st', help='static or dynamic input',type=bool, default=False)
parser.add_argument('--saverates', '-r', help='save rates or not',type=bool, default=False)
args = vars(parser.parse_args())
n_ori = int(args['n_ori'])
n_phs = int(args['n_phs'])
# n_rpt = int(args['n_rpt'])
n_int= int(args['n_int'])
inp_seed = int(args['inp_seed'])
rec_seed = int(args['rec_seed'])
num_noise_seeds = int(args['num_noise_seeds'])
add_phase = args['add_phase']
remove_phase = args['remove_phase']
add_orisel = args['add_orisel']
add_sandp = args['add_sandp']
add_ffl4 = args['add_ffl4']
static = args['static']
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
    
res_dir = res_dir + 'noisy_'

if static:
    res_dir = res_dir + 'static_'
if args['map'] is not None:
    res_dir = res_dir + args['map'] + '_'
if add_phase:
    res_dir = res_dir + 'phase_'
if remove_phase:
    res_dir = res_dir + 'rphase_'
if add_orisel:
    res_dir = res_dir + 'orisel_'
if add_sandp:
    res_dir = res_dir + 'sandp_'
if add_ffl4:
    res_dir = res_dir + 'ffl4_'
res_file = res_dir + 'inp_seed={:d}_rec_seed={:d}.pkl'.format(inp_seed,rec_seed)

res_dict = {}

# load L4 responses
if args['map'] is None:
    with open('./../results/L4_sel/seed={:d}.pkl'.format(inp_seed), 'rb') as handle:
        L4_res_dict = pickle.load(handle)
else:
    with open('./../results/L4_sel/{:s}_seed={:d}.pkl'.format(args['map'],inp_seed), 'rb') as handle:
        L4_res_dict = pickle.load(handle)

if add_ffl4:
    L4_rates = L4_res_dict['L4_rf_rates'][0]
    L4_rate_opm = L4_res_dict['L4_inp_opm'].flatten()
else:
    L4_rates = L4_res_dict['L4_rates'][0]
    L4_rate_opm = L4_res_dict['L4_rate_opm'][0]

L4_rates /= np.nanmean(L4_rates,axis=(-2,-1),keepdims=True)
if add_phase:
    _,_,phs = af.calc_dc_ac_comp(L4_rates)
    L4_phase_rates = np.fmax(0,np.cos(np.linspace(0,2*np.pi,n_phs,endpoint=False)[None,None,:]-phs[:,:,None]))
    L4_phase_rates *= np.nanmean(L4_rates,axis=(-1),keepdims=True) / np.nanmean(L4_phase_rates,axis=(-1),keepdims=True)
    L4_rates = L4_phase_rates
elif remove_phase:
    L4_rates = np.nanmean(L4_rates,axis=(-1),keepdims=True) * np.ones_like(L4_rates)
if add_orisel:
    _,_,doub_po = af.calc_dc_ac_comp(L4_rates.mean(-1))
    L4_orisel_rates = np.fmax(0,np.cos(np.linspace(0,2*np.pi,n_ori,endpoint=False)[None,:]-doub_po[:,None]))
    L4_orisel_rates *= np.nanmean(L4_rates.mean(-1),axis=(-1),keepdims=True) / np.nanmean(L4_orisel_rates,axis=(-1),keepdims=True)
    L4_norm_phase_tuning = np.fmax(1e-12,L4_rates / np.nanmean(L4_rates,axis=(-1),keepdims=True))
    L4_rates = L4_norm_phase_tuning * L4_orisel_rates[:,:,None]
if add_sandp:
    rng = np.random.default_rng(inp_seed)
    L4_rates = rng.permutation(L4_rates)

# Compute distance matrix for connectivity kernel
xs,ys = np.meshgrid(np.arange(N)/N,np.arange(N)/N)
dxs = np.abs(xs[:,:,None,None] - xs[None,None,:,:])
dxs[dxs > 0.5] = 1 - dxs[dxs > 0.5]
dys = np.abs(ys[:,:,None,None] - ys[None,None,:,:])
dys[dys > 0.5] = 1 - dys[dys > 0.5]
dss = np.sqrt(dxs**2 + dys**2).reshape(N**2,N**2)

nbins = 50

idxs = np.digitize(dss,np.linspace(0,np.max(dss),nbins+1))

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
    
# define L4 rate interpolation function after L4 to L2/3 scattering
L4_rates_itp = CubicSpline(np.arange(0,n_phs+1) * 1/(3*n_phs),
                           np.concatenate((L4_rates,L4_rates[:,:,0:1]),axis=-1),
                           axis=-1,bc_type='periodic')

def gen_corr_inps(rng,T=3,dt=0.05/(3*n_phs),t_corr=0.02,noise_cv=5):
    npatt = int(np.round(T/dt)) + 1

    spat_freq = 8

    patts_fft = np.fft.fft2(rng.normal(size=(npatt,N,N)))
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

    for i in range(npatt-1):
        patts[i+1] = patts[i]*np.exp(-dt/t_corr) + patts[i+1]*np.sqrt((1-np.exp(-2*dt/t_corr)))
        patts[i+1] /= np.std(patts[i+1])
        
    gam_dist = gamma(a=1/(noise_cv**2),scale=noise_cv**2)
    
    return PchipInterpolator(np.arange(npatt) * dt,gam_dist.ppf(norm.cdf(patts)))

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

def get_sheet_resps(params,N,noise_itp,noise_str=0.3):
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
    
    nori = n_ori
    nphs = n_phs
    nwrm = 9 * nphs
    dt = 1 / (nphs * 3)
    
    kern_e = np.exp(-(dss/params[4])**2)
    norm = kern_e.sum(axis=1).mean(axis=0)
    kern_e /= norm
    
    kern_i = np.exp(-(dss/params[5])**2)
    norm = kern_i.sum(axis=1).mean(axis=0)
    kern_i /= norm
    
    thresh_e = -params[7]
    thresh_i = -params[8]
    
    tsamp = np.arange(6*nphs,nwrm)
    resps = np.zeros((2,N**2,nori,len(tsamp)))
    for ori_idx in range(nori):
        if static:
            for phs_idx,phs in enumerate(np.linspace(0,2*np.pi,n_phs,endpoint=False)):
                def ff_inp(t):
                    return L4_rates_itp(phs/(2*np.pi*3))[:,ori_idx] + noise_str*noise_itp(t)
                resps[:,:,ori_idx,phs_idx] = integrate_sheet(np.zeros(N**2),np.zeros(N**2),np.zeros(N**2),
                                        np.zeros(N**2),np.zeros(N**2),np.zeros(N**2),
                                        ff_inp,Jee,Jei,Jie,Jii,kern_e,kern_i,params[6],N,2,2,
                                        thresh_e,thresh_i,0,dt,nwrm/2,tsamp[0:1]/2)[:,:,-1]
        else:
            def ff_inp(t):
                return L4_rates_itp(t)[:,ori_idx] + noise_str*noise_itp(t)
            resps[:,:,ori_idx,:] = integrate_sheet(np.zeros(N**2),np.zeros(N**2),np.zeros(N**2),
                                    np.zeros(N**2),np.zeros(N**2),np.zeros(N**2),
                                    ff_inp,Jee,Jei,Jie,Jii,kern_e,kern_i,params[6],N,2,2,
                                    thresh_e,thresh_i,0,dt,nwrm,tsamp)
    return resps

# Integrate to get firing rates
start = time.process_time()

L23_rates = np.zeros((num_noise_seeds,3,2,N**2,n_ori,n_phs))
for noise_seed in tqdm(range(num_noise_seeds)):
    noise_itp = gen_corr_inps(np.random.default_rng(noise_seed))
    L23_rates[noise_seed] = get_sheet_resps(params,N,noise_itp).reshape((2,N**2,n_ori,3,n_phs)).transpose(3,0,1,2,4)

print('Simulating rate dynamics took',time.process_time() - start,'s\n')

if saverates:
    res_dict['L23_rates'] = L23_rates

# Calculate CV of inputs and responses
L23_rate_r0 = np.mean(L23_rates,(0,1,-2,-1))
L23_rate_opm,L23_rate_mr = af.calc_OPM_MR(L23_rates.mean((0,1)))
L23_rate_r1 = np.abs(L23_rate_opm)*L23_rate_r0
L23_inp_opm,L23_inp_mr = af.calc_OPM_MR(L4_rates**2)

res_dict['L23_rate_r0'] = L23_rate_r0
res_dict['L23_rate_r1'] = L23_rate_r1
res_dict['L23_rate_opm'] = L23_rate_opm
res_dict['L23_rate_mr'] = L23_rate_mr
res_dict['L23_inp_opm'] = L23_inp_opm
res_dict['L23_inp_mr'] = L23_inp_mr

# Calculate hypercolumn size and number of pinwheels
_,L23_rate_raps = af.get_fps(L23_rate_opm[0].reshape(N,N))
L23_rate_hc,_ = af.calc_hypercol_size(L23_rate_raps,N)
freqs = np.arange(len(L23_rate_raps))/60
pwd,popt = af.calc_pinwheel_density_from_raps(freqs,L23_rate_raps,return_fit=True)

Lam = L23_rate_hc

res_dict['L23_rate_raps'] = L23_rate_raps
res_dict['L23_rate_hc'] = L23_rate_hc
res_dict['L23_rate_pwd'] = pwd

# Calculate orientation mismatch
L23_rate_pref_ori = np.angle(L23_rate_opm)*180/(2*np.pi)
L23_rate_pref_ori[L23_rate_pref_ori > 90] -= 180
L4_rate_pref_ori = np.angle(L4_rate_opm)*180/(2*np.pi)
L4_rate_pref_ori[L4_rate_pref_ori > 90] -= 180
opm_mismatch = np.abs(L4_rate_pref_ori - L23_rate_pref_ori)
opm_mismatch[opm_mismatch > 90] = 180 - opm_mismatch[opm_mismatch > 90]

res_dict['opm_mismatch'] = opm_mismatch
res_dict['E_mismatch'] = np.mean(opm_mismatch[0])
res_dict['I_mismatch'] = np.mean(opm_mismatch[1])

with open(res_file, 'wb') as handle:
    pickle.dump(res_dict,handle)
