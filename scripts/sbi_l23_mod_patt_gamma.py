import os
import pickle
import time
import argparse

import numpy as np
import torch
from scipy import interpolate
from scipy import integrate
from scipy.signal import argrelmin,argrelmax
from scipy.stats import norm,gamma

from sbi.utils.user_input_checks import process_prior
from sbi.utils import BoxUniform

import analyze_func as af
import map_func as mf
from sbi_func import PostTimesBoxUniform

parser = argparse.ArgumentParser()
parser.add_argument('--job_id', '-i', help='completely arbitrary job id label',type=int, default=0)
parser.add_argument('--num_samp', '-ns', help='number of samples',type=int, default=100)
parser.add_argument('--bayes_iter', '-bi', help='bayessian inference interation (0 = use prior, 1 = use first posterior)',type=int, default=0)
args = vars(parser.parse_args())
job_id = int(args['job_id'])
num_samp = int(args['num_samp'])
bayes_iter = int(args['bayes_iter'])

print("Bayesian iteration:", bayes_iter)
print("Job ID:", job_id)

device = torch.device("cpu")

# Define where to save results
res_dir = './../results/'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

res_dir = res_dir + 'sbi_l23_mod_patt_gamma/'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

res_file = res_dir + 'bayes_iter={:d}_job={:d}.pkl'.format(bayes_iter, job_id)

# create prior distribution
if bayes_iter == 0:
    prior = BoxUniform(low =torch.tensor([ 0.0,-2.0,-2.0,-2.0, 0.02, 0.02, 0.01],device=device),
                       high=torch.tensor([ 1.0, 2.0, 1.0, 1.0, 0.1 , 0.1 , 0.5 ],device=device),)

    # prior,_,_ = process_prior(prior)
    # with open(f'./../notebooks/l23_patt_posterior_2.pkl','rb') as handle:
    #    prior = pickle.load(handle)
else:
    with open(f'./../notebooks/l23_patt_gamma_posterior_{bayes_iter:d}.pkl','rb') as handle:
        prior = pickle.load(handle)

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

npatt = 50
patts_fft = np.fft.fft2(np.random.default_rng(0).normal(size=(npatt,N,N)))
patts_fft[:,0,0] = 0 # remove DC component
freqs = np.fft.fftfreq(N,1/N)
freqs = np.sqrt(freqs[:,None]**2 + freqs[None,:]**2)

decay = 12
patts_fft *= np.exp(-0.5*freqs**2/decay**2)[None,:,:]

patts = np.real(np.fft.ifft2(patts_fft).reshape(npatt,-1))
for i in range(10):
    patts -= np.mean(patts,axis=-1,keepdims=True)
    patts /= np.std(patts,axis=-1,keepdims=True)
    
    patts -= np.mean(patts,axis=0,keepdims=True)
    patts /= np.std(patts,axis=0,keepdims=True)
    
patt_cv = 1.2
gam_dist = gamma(a=1/(patt_cv**2),scale=patt_cv**2)
patts = gam_dist.ppf(norm.cdf(patts))
    
dim_inp = 46.58640395399712

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

def get_sheet_resps(theta,N):
    Jee,Jei,Jie,Jii = get_J(theta)
    
    thresh = 0
    nint = 3
    nwrm = 15 * nint
    dt = 0.01 / nint
    
    tsamp = np.array([nwrm-1])
    resps = np.zeros((theta.shape[0],2,N**2,npatt))
    for prm_idx in range(theta.shape[0]):
        kern_e = np.exp(-(dss/(theta[prm_idx,4].item()))**2)
        norm = kern_e.sum(axis=1).mean(axis=0)
        kern_e /= norm
        
        kern_i = np.exp(-(dss/(theta[prm_idx,5].item()))**2)
        norm = kern_i.sum(axis=1).mean(axis=0)
        kern_i /= norm
        
        for patt_idx,patt in enumerate(patts):
            def ff_inp(t):
                return patt
            resp = integrate_sheet(np.zeros(N**2),np.zeros(N**2),np.zeros(N**2),
                                    np.zeros(N**2),np.zeros(N**2),np.zeros(N**2),
                                    ff_inp,Jee[prm_idx].item(),Jei[prm_idx].item(),
                                    Jie[prm_idx].item(),Jii[prm_idx].item(),
                                    kern_e,kern_i,theta[prm_idx,6].item(),N,2,2,
                                    thresh,thresh,0,dt,nwrm,tsamp)
            resps[prm_idx,:,:,patt_idx] = resp.transpose((2,0,1,3))[:,:,:,0]
        
    return resps

def sheet_simulator(theta):
    '''
    theta[:,0] = det(J)/(|Jei| * |Jie|) = 1 - (|Jee| * |Jii|) / (|Jei| * |Jie|)
    theta[:,1] = (|Jee|-|Jii|)/(|Jei| + |Jie|)
    theta[:,2] = (log10[|Jei|] + log10[|Jie|]) / 2
    theta[:,3] = (log10[|Jei|] - log10[|Jie|]) / 2
    theta[:,4] = s_e
    theta[:,5] = s_i
    theta[:,6] = het_level
    theta[:,7] = inp_str
    
    returns: [mod,corr_min,corr_max,freq,dim,min_r]
    mod = excitatory response modularity
    corr_min = excitatory response correlation first minimum
    corr_max = excitatory response correlation first maximum after the minimum
    freq = spatial frequency corresponding to corr_max
    dim = excitatory response dimensionality
    min_r = average minimum excitatory response relative to the maximum
    '''
    
    resps = get_sheet_resps(theta,N)
    
    resp_z = resps[:,0,:,:]
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
            # w = np.linalg.eigvalsh(corr[i,:,:])
            # dim[i] = np.sum(w)**2/np.sum(w**2)
            dim[i] = np.trace(corr[i,:,:])**2 / np.trace(corr[i,:,:] @ corr[i,:,:])
        except:
            dim[i] = dim_inp
    dim /= dim_inp
    
    min_r = np.mean(np.min(resps[:,0,:,:],axis=-2) / np.max(resps[:,0,:,:],axis=-2),axis=-1)
    
    out = torch.zeros((theta.shape[0],6),dtype=theta.dtype).to(theta.device)
    out[:,0] = torch.tensor(mod,dtype=theta.dtype).to(theta.device)
    out[:,1] = torch.tensor(corr_mins,dtype=theta.dtype).to(theta.device)
    out[:,2] = torch.tensor(corr_maxs,dtype=theta.dtype).to(theta.device)
    out[:,3] = torch.tensor(arg_maxs,dtype=theta.dtype).to(theta.device)
    out[:,4] = torch.tensor(dim,dtype=theta.dtype).to(theta.device)
    out[:,5] = torch.tensor(min_r,dtype=theta.dtype).to(theta.device)
    
    valid_idx = torch.all(torch.tensor(resps) < 5e4,axis=(1,2,3))
    
    return torch.where(valid_idx[:,None],out,torch.tensor([torch.nan])[:,None])

start = time.process_time()

thetas = torch.zeros((0,7),dtype=torch.float32,device=device)
xs = torch.zeros((0,6),dtype=torch.float32,device=device)

while thetas.shape[0] < num_samp:
    this_samps = 1
    
    start = time.process_time()
    # sample from prior
    theta = prior.sample((this_samps,))
    # simulate sheet
    x = sheet_simulator(theta)

    thetas = torch.cat([thetas,theta],dim=0)
    xs = torch.cat([xs,x],dim=0)

    print(f'Simulating samples took',time.process_time() - start,'s\n')

    # save results
    with open(res_file, 'wb') as handle:
        pickle.dump({
            'theta': thetas,
            'x': xs,
        }, handle)
