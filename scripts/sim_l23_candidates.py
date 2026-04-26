import os
import pickle
import time
import argparse

import numpy as np
import torch
from scipy.interpolate import CubicSpline
from scipy import integrate
from scipy.signal import argrelmin,argrelmax
from scipy.stats import norm,gamma

import analyze_func as af

parser = argparse.ArgumentParser()
parser.add_argument('--batch_iter', '-bi', help='initial candidate',type=int, default=0)
parser.add_argument('--per_batch', '-pb', help='number candidates per run',type=int, default=1)
args = vars(parser.parse_args())
batch_iter = int(args['batch_iter'])
per_batch = int(args['per_batch'])

print("Running candidates:", range(batch_iter*per_batch,(batch_iter+1)*per_batch))
device = torch.device("cpu")

# Define where to save results
res_dir = './../results/'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

res_dir = res_dir + 'sim_l23_candidates/'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

res_file = res_dir + 'candidate={:d}-{:d}.pkl'.format(init_iter, init_iter+batch_iter-1)

with open('./../notebooks/l23_candidate_prms.pkl', 'rb') as handle:
    candidate_prms = pickle.load(handle)

# load L4 responses
def load_l4_rates(file):
    with open(file, 'rb') as handle:
        L4_res_dict = pickle.load(handle)
        L4_rates = L4_res_dict['L4_rates'][0]
        L4_rate_opm = L4_res_dict['L4_rate_opm'][0]
    L4_rates /= np.nanmean(L4_rates,axis=(-2,-1),keepdims=True)

    L4_rates_itp = CubicSpline(np.arange(0,8+1) * 1/(3*8),
                            np.concatenate((L4_rates,L4_rates[:,:,0:1]),axis=-1),
                            axis=-1,bc_type='periodic')
    
    return L4_rates_itp, L4_rate_opm

# create distances between grid points
N = 60

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

# define simulation functions
def integrate_sheet(xea0,xen0,xeg0,xia0,xin0,xig0,inp,Jee,Jei,Jie,Jii,kern_e,kern_i,het_lev,N,ne,ni,threshe,threshi,
                    t0,dt,Nt,tsamp=None,ta=0.01,tn=0.300,tg=0.01,frac_n=0.7):
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
    
    if tsamp is None:
        tsamp = [Nt-1]
    samp_idx = 0
    
    xea = xea0.copy()
    xen = xen0.copy()
    xeg = xeg0.copy()
    xia = xia0.copy()
    xin = xin0.copy()
    xig = xig0.copy()
    
    rng = np.random.default_rng(0)
    
    if np.isscalar(Jee):
        gam_dist = gamma(a=1/(het_lev**2),scale=het_lev**2)
        
        Wee = Jee*kern_e.reshape(N**2,N**2)*gam_dist.ppf(norm_dist.cdf(gen_noise(rng)))
        Wei = Jei*kern_i.reshape(N**2,N**2)*gam_dist.ppf(norm_dist.cdf(gen_noise(rng)))
        Wie = Jie*kern_e.reshape(N**2,N**2)*gam_dist.ppf(norm_dist.cdf(gen_noise(rng)))
        Wii = Jii*kern_i.reshape(N**2,N**2)*gam_dist.ppf(norm_dist.cdf(gen_noise(rng)))
        
        Wee = Wee[:,:,None]
        Wei = Wei[:,:,None]
        Wie = Wie[:,:,None]
        Wii = Wii[:,:,None]
        
        if len(xea.shape) == 1:
            xea = xea[:,None]
            xen = xen[:,None]
            xeg = xeg[:,None]
            xia = xia[:,None]
            xin = xin[:,None]
            xig = xig[:,None]
            
        nprm = 1
        resps = np.zeros((2,N**2,1,len(tsamp)))
    else:
        Wee = Jee[None,None,:]*kern_e.reshape(N**2,N**2,-1)*(1+het_lev[None,None,:]*gen_noise(rng)[:,:,None])
        Wei = Jei[None,None,:]*kern_i.reshape(N**2,N**2,-1)*(1+het_lev[None,None,:]*gen_noise(rng)[:,:,None])
        Wie = Jie[None,None,:]*kern_e.reshape(N**2,N**2,-1)*(1+het_lev[None,None,:]*gen_noise(rng)[:,:,None])
        Wii = Jii[None,None,:]*kern_i.reshape(N**2,N**2,-1)*(1+het_lev[None,None,:]*gen_noise(rng)[:,:,None])
        
        if len(xea.shape) == 1:
            xea = xea[:,None] * np.ones(len(Jee))[None,:]
            xen = xen[:,None] * np.ones(len(Jee))[None,:]
            xeg = xeg[:,None] * np.ones(len(Jee))[None,:]
            xia = xia[:,None] * np.ones(len(Jee))[None,:]
            xin = xin[:,None] * np.ones(len(Jee))[None,:]
            xig = xig[:,None] * np.ones(len(Jee))[None,:]
        
        nprm = len(Jee)
        resps = np.zeros((2,N**2,len(Jee),len(tsamp)))
        
    def dyn_func(t,x,ncell,nprm=1):
        x = x.reshape((-1,nprm))
        xea = x[0*ncell:1*ncell,:]
        xen = x[1*ncell:2*ncell,:]
        xeg = x[2*ncell:3*ncell,:]
        xia = x[3*ncell:4*ncell,:]
        xin = x[4*ncell:5*ncell,:]
        xig = x[5*ncell:6*ncell,:]
        
        ff_inp = inp(t)

        ye = np.fmin(1e5,np.fmax(0,xea+xen+xeg-threshe)**ne)
        yi = np.fmin(1e5,np.fmax(0,xia+xin+xig-threshi)**ni)
        
        net_ee = np.einsum('ijk,jk->ik',Wee,ye) + ff_inp[:,None]
        net_ei = np.einsum('ijk,jk->ik',Wei,yi)
        net_ie = np.einsum('ijk,jk->ik',Wie,ye) + ff_inp[:,None]
        net_ii = np.einsum('ijk,jk->ik',Wii,yi)
        
        dx = np.zeros_like(x)
        dx[0*ncell:1*ncell,:] = ((1-frac_n)*net_ee - xea)/ta
        dx[1*ncell:2*ncell,:] = (frac_n*net_ee - xen)/tn
        dx[2*ncell:3*ncell,:] = (net_ei - xeg)/tg
        dx[3*ncell:4*ncell,:] = ((1-frac_n)*net_ie - xia)/ta
        dx[4*ncell:5*ncell,:] = (frac_n*net_ie - xin)/tn
        dx[5*ncell:6*ncell,:] = (net_ii - xig)/tg
        
        return dx.flatten()
    
    x0 = np.concatenate((xea,xen,xeg,xia,xin,xig),axis=0).flatten()
    
    start_time = time.process_time()
    max_time = 60
    def time_event(t,x,ncell,nprm):
        int_time = (start_time + max_time) - time.process_time()
        if int_time < 0: int_time = 0
        return int_time
    time_event.terminal = True
    
    sol = integrate.solve_ivp(dyn_func,(0,dt*Nt),y0=x0,t_eval=tsamp*dt,args=(N**2,nprm),method='RK23')#,events=time_event)
    if sol.status != 0:
        x = np.nan*np.ones((6*N**2*nprm,len(tsamp)))
    else:
        x = sol.y
    x = x.reshape((-1,nprm,len(tsamp)))
    
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
    
    return resps

def get_J(theta):
    '''
    theta[:,0] = det(J)/(|Jei| * |Jie|) = 1 - (|Jee| * |Jii|) / (|Jei| * |Jie|)
    theta[:,1] = (|Jee|-|Jii|)/(|Jei| + |Jie|)
    theta[:,2] = (log10[|Jei|] + log10[|Jie|]) / 2
    theta[:,3] = (log10[|Jei|] - log10[|Jie|]) / 2
    
    returns: [Jee,Jei,Jie,Jii]
    '''
    Jei = -10**(theta[:,2] + theta[:,3])
    Jie =  10**(theta[:,2] - theta[:,3])
    Jee_m_Jii = (-Jei + Jie) * theta[:,1]
    Jee_p_Jii_2 = 4*((theta[:,0] - 1) * Jei*Jie) + Jee_m_Jii**2
    Jee = 0.5*(Jee_m_Jii + torch.sqrt(Jee_p_Jii_2))
    Jii = -(Jee - Jee_m_Jii)
    
    return Jee,Jei,Jie,Jii

def get_sheet_resps(theta,N,L4_rates_itp):
    Jee,Jei,Jie,Jii = get_J(theta)
    Jee *= 10**theta[:,9]
    Jei *= 10**theta[:,9]
    Jie *= 10**theta[:,9]
    Jii *= 10**theta[:,9]
    
    nori = 8
    nphs = 8
    nint = 5
    nwrm = 6 * nint * nphs
    dt = 1 / (nint * nphs * 3)
    
    tsamp = nwrm-1 + np.arange(0,nphs) * nint
    resps = np.zeros((theta.shape[0],2,N**2,nori,nphs))
    
    for prm_idx in range(theta.shape[0]):
        kern_e = np.exp(-(dss/(theta[prm_idx,4].item()))**2)
        norm = kern_e.sum(axis=1).mean(axis=0)
        kern_e /= norm
        
        kern_i = np.exp(-(dss/(theta[prm_idx,5].item()))**2)
        norm = kern_i.sum(axis=1).mean(axis=0)
        kern_i /= norm
        
        thresh_e = -theta[prm_idx,7].item()
        thresh_i = -theta[prm_idx,8].item()
        
        for ori_idx in range(nori):
            def ff_inp(t):
                return L4_rates_itp(t)[:,ori_idx]
            resp = integrate_sheet(np.zeros(N**2),np.zeros(N**2),np.zeros(N**2),
                                    np.zeros(N**2),np.zeros(N**2),np.zeros(N**2),
                                    ff_inp,Jee[prm_idx].item(),Jei[prm_idx].item(),
                                    Jie[prm_idx].item(),Jii[prm_idx].item(),
                                    kern_e,kern_i,theta[prm_idx,6].item(),N,2,2,
                                    thresh_e,thresh_i,0,dt,nwrm+nint*nphs,tsamp)
            resps[prm_idx,:,:,ori_idx,:] = resp.transpose((2,0,1,3))
        
    return resps

def sheet_simulator(theta,file):
    '''
    theta[:,0] = det(J)/(|Jei| * |Jie|) = 1 - (|Jee| * |Jii|) / (|Jei| * |Jie|)
    theta[:,1] = (|Jee|-|Jii|)/(|Jei| + |Jie|)
    theta[:,2] = (log10[|Jei|] + log10[|Jie|]) / 2
    theta[:,3] = (log10[|Jei|] - log10[|Jie|]) / 2
    theta[:,4] = s_e
    theta[:,5] = s_i
    theta[:,6] = het_level
    theta[:,7] = base_e
    theta[:,8] = base_i
    theta[:,9] = log10[J_mult]
    
    returns: [q1_os,q2_os,q3_os,mu_os,sig_os,q1_mr,q2_mr,q3_mr,mu_mr,sig_mr,mu_mm,pwd,mod,corr_min,corr_max,freq,dim]
    os = excitatory orientation selectivity
    mr = excitatory modulation ratio
    mm = input-output mismatch
    pwd = pinwheel density
    mod = excitatory response modularity
    corr_min = excitatory response correlation first minimum
    corr_max = excitatory response correlation first maximum after the minimum
    freq = spatial frequency corresponding to corr_max
    dim = excitatory response dimensionality
    '''
    
    L4_rates_itp, L4_rate_opm = load_l4_rates(file)
    
    resps = get_sheet_resps(theta,N,L4_rates_itp)
    
    opm,mr = af.calc_OPM_MR(resps[:,0,:,:,:])
    os = np.abs(opm)
    
    inp_po = np.angle(L4_rate_opm)*180/(2*np.pi)
    inp_po[inp_po > 90] -= 180
    out_po = np.angle(opm)*180/(2*np.pi)
    out_po[out_po > 90] -= 180

    mm = np.abs(inp_po - out_po)
    mm[mm > 90] = 180 - mm[mm > 90]
    
    _,raps = af.get_fps(opm.reshape(-1,N,N))
    freqs = np.zeros(theta.shape[0],dtype=int)
    for i in range(theta.shape[0]):
        try:
            freqs[i] = np.argmax(raps[i])
        except:
            freqs[i] = np.nan
    pwd = af.calc_pinwheel_density_from_raps(np.arange(raps.shape[-1])[None,:]/N,
                                             raps,continuous=True)
    
    resp_z = resps[:,0,:,:,:].reshape(theta.shape[0],N**2,-1)
    npatt = resp_z.shape[-1]
    resp_z = resp_z - np.mean(resp_z,axis=-1,keepdims=True)
    resp_z = resp_z / np.std(resp_z,axis=-1,keepdims=True)
    corr = np.zeros((theta.shape[0],N**2,N**2))
    for i in range(npatt):
        corr += resp_z[:,None,:,i] * resp_z[:,:,None,i]
    corr /= npatt
    
    corr_curve = np.zeros((theta.shape[0],nbins))
    for i in range(nbins):
        corr_curve[:,i] = np.mean(corr[:,idxs == i+1],axis=-1)
    arg_mins = np.zeros(theta.shape[0],dtype=int)
    for i in range(theta.shape[0]):
        try:
            arg_mins[i] = argrelmin(corr_curve[i])[0][0]
        except:
            arg_mins[i] = np.argmin(corr_curve[i])
    corr_mins = np.array([corr_curve[i,arg_mins[i]] for i in range(theta.shape[0])])
    arg_maxs = np.zeros(theta.shape[0],dtype=int)
    for i in range(theta.shape[0]):
        try:
            loc_maxs = argrelmax(corr_curve[i])[0]
            arg_maxs[i] = loc_maxs[0]
            if arg_maxs[i] < arg_mins[i]:
                arg_maxs[i] = loc_maxs[1]
        except:
            arg_maxs[i] = np.argmax(corr_curve[i,arg_mins[i]:]) + arg_mins[i]
    corr_maxs = np.array([corr_curve[i,arg_maxs[i]] for i in range(theta.shape[0])])
    mod = corr_maxs - corr_mins
    
    dim = np.zeros(theta.shape[0])
    for i in range(theta.shape[0]):
        try:
            dim[i] = np.trace(corr[i,:,:])**2 / np.trace(corr[i,:,:] @ corr[i,:,:])
        except:
            dim[i] = npatt
    dim /= npatt
    
    mean_act = resps[:,0,:,:,:].mean((1,2))
    var_t = (np.max(mean_act,axis=-1)/np.min(mean_act,axis=-1) - 1)
    
    var_r = (np.max(raps,axis=-1)/np.mean(raps,axis=-1) - 1)
    
    out = torch.zeros((theta.shape[0],19),dtype=theta.dtype).to(theta.device)
    out[:,0:3] = torch.tensor(np.quantile(os,[0.25,0.50,0.75],axis=1).T,dtype=theta.dtype).to(theta.device)
    out[:,3] = torch.tensor(np.mean(os,axis=1),dtype=theta.dtype).to(theta.device)
    out[:,4] = torch.tensor(np.std(os,axis=1),dtype=theta.dtype).to(theta.device)
    out[:,5:8] = torch.tensor(np.quantile(mr,[0.25,0.50,0.75],axis=1).T,dtype=theta.dtype).to(theta.device)
    out[:,8] = torch.tensor(np.mean(mr,axis=1),dtype=theta.dtype).to(theta.device)
    out[:,9] = torch.tensor(np.std(mr,axis=1),dtype=theta.dtype).to(theta.device)
    out[:,10] = torch.tensor(np.mean(mm,axis=1),dtype=theta.dtype).to(theta.device)
    out[:,11] = torch.tensor(pwd,dtype=theta.dtype).to(theta.device)
    out[:,12] = torch.tensor(mod,dtype=theta.dtype).to(theta.device)
    out[:,13] = torch.tensor(corr_mins,dtype=theta.dtype).to(theta.device)
    out[:,14] = torch.tensor(corr_maxs,dtype=theta.dtype).to(theta.device)
    out[:,15] = torch.tensor(freqs,dtype=theta.dtype).to(theta.device)
    out[:,16] = torch.tensor(dim,dtype=theta.dtype).to(theta.device)
    out[:,17] = torch.tensor(var_t,dtype=theta.dtype).to(theta.device)
    out[:,18] = torch.tensor(var_r,dtype=theta.dtype).to(theta.device)
    
    return out[0], raps[0], corr_curve[0]

xs = torch.zeros((per_batch,2,19),dtype=torch.float32,device=device)
raps = np.zeros((per_batch,2,int(np.round(np.ceil(N//2*np.sqrt(2))))))
corr_curves = np.zeros((per_batch,2,nbins),dtype=torch.float32,device=device)

init_iter = batch_iter * per_batch
for i in range(per_batch):
    this_theta = candidate_prms[init_iter+i:init_iter+i+1].to(device)
    xs[i,0,:], raps[i,0,:], corr_curves[i,0,:] = sheet_simulator(this_theta, './../results/L4_sel/seed=0.pkl')
    xs[i,1,:], raps[i,1,:], corr_curves[i,1,:] = sheet_simulator(this_theta, './../results/L4_sel/band_seed=0.pkl')

res_dict = {'x': xs, 'raps': raps, 'corr_curves': corr_curves}
with open(res_file, 'wb') as handle:
    pickle.dump(res_dict, handle)
