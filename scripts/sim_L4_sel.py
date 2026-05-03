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
parser.add_argument('--seed', '-s', help='seed',type=int, default=0)
parser.add_argument('--saverates', '-r', help='save rates or not',type=bool, default=False)
args = vars(parser.parse_args())
n_ori = int(args['n_ori'])
n_phs = int(args['n_phs'])
# n_rpt = int(args['n_rpt'])
n_int= int(args['n_int'])
static = args['static']
seed = int(args['seed'])
saverates = args['saverates']

N = 60

# Define parameters for connectivity
params = np.load("./../notebooks/l4_params.npy")

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
    res_file = res_dir + 'seed={:d}.pkl'.format(seed)
else:
    res_file = res_dir + '{:s}_seed={:d}.pkl'.format(args['map'],seed)

res_dict = {}

# Create L4 input map
rng = np.random.default_rng(seed)
opm_fft = rng.normal(size=(N,N)) + 1j * rng.normal(size=(N,N))
opm_fft[0,0] = 0 # remove DC component
freqs = np.fft.fftfreq(N,1/N)
freqs = np.sqrt(freqs[:,None]**2 + freqs[None,:]**2)

if args['map'] is None or args['map'] == 'low':
    decay = 5
    opm_fft *= np.exp(-freqs/decay)
elif 'band' in args['map']:
    if args['map'] == 'band':
        peak = 6
    else:
        _,peak = args['map'].split('_')
        peak = float(peak)
    opm_fft *= np.exp(-((freqs-peak)/2.5)**2)#np.heaviside(0.5 - np.abs(freqs-peak),0.5)

L4_inp_opm = np.fft.ifft2(opm_fft)
L4_inp_opm *= np.abs(L4_inp_opm)**1.6/np.abs(L4_inp_opm)
L4_inp_opm *= 0.16 / np.median(np.abs(L4_inp_opm)) # normalize median to data
L4_inp_opm *= np.clip(np.abs(L4_inp_opm),0,0.8) / np.abs(L4_inp_opm) # clip max os to 0.8
if args['map'] == 'sandp':
    L4_inp_opm = L4_inp_opm.flatten()
    rng.shuffle(L4_inp_opm)
    L4_inp_opm = L4_inp_opm.reshape(N,N)

# compute elongation of each rf
oris = np.linspace(0,np.pi,100,endpoint=False) - np.pi/2
phss = np.linspace(0,2*np.pi,100,endpoint=False) - np.pi

kl2 = 2

if static:
    def elong_inp(gam,ori,phs):
        return 1 + 0.4*np.cos(phs)*np.exp(-kl2*(1+(1-gam**2)/gam**2*np.sin(ori)**2)/2)
else:
    def elong_inp(gam,ori,phs):
        return 1 + np.cos(phs)*np.exp(-kl2*(1+(1-gam**2)/gam**2*np.sin(ori)**2)/2)

gams = np.linspace(0.4,1,301)
resps = np.fmax(0,elong_inp(gams[:,None,None],np.linspace(0,np.pi,36,endpoint=False)[None,:,None],np.linspace(0,2*np.pi,36,endpoint=False)[None,None,:])-1)**2
oss,_ = af.calc_OS_MR(resps)

gam_os_itp = interp1d(oss,gams,fill_value='extrapolate')

gam_map = gam_os_itp(np.abs(L4_inp_opm))

# compute rf scatter and ON/OFF bias maps
sig2 = 0.00095

rf_sct_scale = 0.8
pol_scale = np.array([10,5,5])
L_mm = N/11
mag_fact = 0.02
# L_deg = L_mm / np.sqrt(mag_fact)
grate_freq = 0.06
L_deg = 5.9 / grate_freq

sctmap,polmap = mf.gen_rf_sct_map(N,sig2,rf_sct_scale,pol_scale,EI_match=True,kern_type='bandplushighpass',seed=seed)

# Compute distance matrix for connectivity kernel
xs,ys = np.meshgrid(np.arange(N)/N,np.arange(N)/N)
dxs = np.abs(xs[:,:,None,None] - xs[None,None,:,:])
dxs[dxs > 0.5] = 1 - dxs[dxs > 0.5]
dys = np.abs(ys[:,:,None,None] - ys[None,None,:,:])
dys[dys > 0.5] = 1 - dys[dys > 0.5]
dss = np.sqrt(dxs**2 + dys**2).reshape(N**2,N**2)

# define simulation functions
def integrate_sheet(xea0,xen0,xeg0,xia0,xin0,xig0,inp,Jee,Jei,Jie,Jii,kern_n,kern_b,N,ne,ni,threshe,threshi,
                    t0,dt,Nt,tsamp,ta=0.01,tn=0.300,tg=0.01,frac_n=0.7,b_frac_e=1.0,b_frac_i=1.0):
    '''
    Integrate 2D sheet with AMPA, NMDA, and GABA receptor dynamics.
    xe0, xi0: initial excitatory and inhibitory activity
    inp: input function, takes time t and returns input at that time
    Jee, Jei, Jie, Jii: connectivity strengths per connection type
    kern: connectivity kernel for the sheet
    ne, ni: rate activation exponents for excitatory and inhibitory neurons
    threshe, threshi: activation thresholds for excitatory and inhibitory neurons
    t0: initial time
    dt: time step for integration
    Nt: number of time steps to integrate
    ta, tn, tg: time constants for AMPA, NMDA, and GABA receptor dynamics
    frac_n: fraction of NMDA vs NMDA+AMPA receptors in the excitatory population
    '''

    xea = xea0.copy()
    xen = xen0.copy()
    xeg = xeg0.copy()
    xia = xia0.copy()
    xin = xin0.copy()
    xig = xig0.copy()

    Wee = Jee*(kern_n.reshape(N**2,N**2) + b_frac_e*kern_b.reshape(N**2,N**2))
    Wei = Jei*(kern_n.reshape(N**2,N**2) + b_frac_i*kern_b.reshape(N**2,N**2))
    Wie = Jie*(kern_n.reshape(N**2,N**2) + b_frac_e*kern_b.reshape(N**2,N**2))
    Wii = Jii*(kern_n.reshape(N**2,N**2) + b_frac_i*kern_b.reshape(N**2,N**2))

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
    return xea,xen,xeg,xia,xin,xig,np.concatenate((ye,yi))

def get_sheet_rf_resps(N,gam_map,ori_map,rf_sct_map,pol_map):
    gam_map_flat = gam_map.flatten()
    ori_map_flat = ori_map.flatten()

    c = 100
    thresh = c
    oris = np.linspace(0,np.pi,n_ori,endpoint=False)

    resps = np.zeros((2,N**2,n_ori,n_phs))
    for ori_idx,ori in enumerate(oris):
        phs_map_flat = mf.gen_abs_phs_map(N,rf_sct_map,pol_map,ori,grate_freq,L_deg).flatten()
        def ff_inp(t):
            return c*elong_inp(gam_map_flat,ori-ori_map_flat,phs_map_flat+2*np.pi*3*t)
        for phs_idx in range(n_phs):
            resps[:,:,ori_idx,phs_idx] = (np.fmax(0,ff_inp(phs_idx/n_phs / 3)-thresh)**2).reshape(1,N**2)

    return resps

def get_sheet_resps(params,N,gam_map,ori_map,rf_sct_map,pol_map):
    '''
    params[0] = log10[|Jee]]
    params[1] = log10[|Jei]]
    params[2] = log10[|Jie]]
    params[3] = log10[|Jii]]
    params[4] = Jb_e / Jn_e
    params[5] = Jb_i / Jn_i
    params[6] = s_b

    returns: resps, array of shape (theta.shape[0],2,N**2,n_ori=8,n_phs=8)
    '''
    Jee,Jei,Jie,Jii = 10**params[:4]
    Jei *= -1
    Jii *= -1

    c = 100
    thresh = c
    nwrm = 6 * n_int * n_phs
    dt = 1 / (n_int * n_phs * 3)
    oris = np.linspace(0,np.pi,n_ori,endpoint=False)

    s_b = np.sqrt(sig2)*params[6]

    kern_n = np.eye(N**2)

    kern_b = np.exp(-(dss/s_b)**2)
    norm = kern_b.sum(axis=1).mean(axis=0)
    kern_b /= norm

    gam_map_flat = gam_map.flatten()
    ori_map_flat = ori_map.flatten()

    tsamp = nwrm-1 + np.arange(0,n_phs) * n_int
    resps = np.zeros((2,N**2,n_ori,n_phs))
    for ori_idx,ori in enumerate(oris):
        phs_map_flat = mf.gen_abs_phs_map(N,rf_sct_map,pol_map,ori,grate_freq,L_deg).flatten()
        if static:
            for phs_idx,phs in enumerate(np.linspace(0,2*np.pi,n_phs,endpoint=False)):
                def ff_inp(t):
                    return c*elong_inp(gam_map_flat,ori-ori_map_flat,phs_map_flat+phs)
                _,_,_,_,_,_,resp = integrate_sheet(np.zeros(N**2),np.zeros(N**2),np.zeros(N**2),
                                        np.zeros(N**2),np.zeros(N**2),np.zeros(N**2),
                                        ff_inp,Jee,Jei,Jie,Jii,kern_n,kern_b,N,2,2,
                                        thresh,thresh,0,dt,nwrm,tsamp[0:1],
                                        b_frac_e=params[4],b_frac_i=params[5])
                resps[:,:,ori_idx,phs_idx] = resp.reshape(2,N**2)

        else:
            def ff_inp(t):
                return c*elong_inp(gam_map_flat,ori-ori_map_flat,phs_map_flat+2*np.pi*3*t)
            _,_,_,_,_,_,resp = integrate_sheet(np.zeros(N**2),np.zeros(N**2),np.zeros(N**2),
                                    np.zeros(N**2),np.zeros(N**2),np.zeros(N**2),
                                    ff_inp,Jee,Jei,Jie,Jii,kern_n,kern_b,N,2,2,
                                    thresh,thresh,0,dt,nwrm+n_int*n_phs,tsamp,
                                    b_frac_e=params[4],b_frac_i=params[5])
            resps[:,:,ori_idx,:] = resp.reshape(2,N**2,n_phs)
        # xea,xen,xeg,xia,xin,xig,resp = integrate_sheet(np.zeros(N**2),np.zeros(N**2),np.zeros(N**2),
        #                          np.zeros(N**2),np.zeros(N**2),np.zeros(N**2),
        #                          ff_inp,Jee,Jei,Jie,Jii,kern,N,2,2,
        #                          thresh,thresh,0,dt,nwrm,lat_frac=params[4])
        # resps[:,:,ori_idx,0] = resp.reshape(2,N**2)
        # for phs_idx in range(n_phs-1):
        #     xea,xen,xeg,xia,xin,xig,resp = integrate_sheet(xea,xen,xeg,xia,xin,xig,
        #                          ff_inp,Jee,Jei,Jie,Jii,kern,N,2,2,
        #                          thresh,thresh,phs_idx*n_int*dt,dt,n_int,lat_frac=params[4])
        #     resps[:,:,ori_idx,phs_idx+1] = resp.reshape(2,N**2)

    return resps

# Integrate to get firing rates
start = time.process_time()

L4_rf_rates = get_sheet_rf_resps(N,gam_map,np.angle(L4_inp_opm)/2,sctmap,polmap)
L4_rates = get_sheet_resps(params,N,gam_map,np.angle(L4_inp_opm)/2,sctmap,polmap)

print('Simulating rate dynamics took',time.process_time() - start,'s\n')

if saverates:
    res_dict['L4_rf_rates'] = L4_rf_rates
    res_dict['L4_rates'] = L4_rates

# Calculate CV of inputs and responses
inp_r0 = np.mean(L4_rf_rates,(-2,-1))
inp_opm,inp_mr = af.calc_OPM_MR(L4_rf_rates)
# inp_os = np.abs(inp_opm)
# inp_po = np.angle(inp_opm)*180/(2*np.pi)
inp_r1 = np.abs(inp_opm)*inp_r0

L4_rate_r0 = np.mean(L4_rates,(-2,-1))
L4_rate_opm,L4_rate_mr = af.calc_OPM_MR(L4_rates)
# L4_rate_os = np.abs(L4_rate_opm)
# L4_rate_po = np.angle(L4_rate_opm)*180/(2*np.pi)
L4_rate_r1 = np.abs(L4_rate_opm)*L4_rate_r0

res_dict['L4_inp_opm'] = L4_inp_opm
res_dict['inp_r0'] = inp_r0
res_dict['inp_r1'] = inp_r1
res_dict['inp_opm'] = inp_opm
res_dict['inp_mr'] = inp_mr

res_dict['L4_rate_r0'] = L4_rate_r0
res_dict['L4_rate_r1'] = L4_rate_r1
res_dict['L4_rate_opm'] = L4_rate_opm
res_dict['L4_rate_mr'] = L4_rate_mr

with open(res_file, 'wb') as handle:
    pickle.dump(res_dict,handle)
