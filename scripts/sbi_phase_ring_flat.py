import os
import pickle
import time
import argparse

import numpy as np
import torch
from scipy import integrate

from sbi.utils import BoxUniform
from sbi.utils.user_input_checks import process_prior

import analyze_func as af

parser = argparse.ArgumentParser()
parser.add_argument('--job_id', '-i', help='completely arbitrary job id label',type=int, default=0)
parser.add_argument('--num_samp', '-ns', help='number of samples',type=int, default=2000)
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

res_dir = res_dir + 'sbi_phase_ring_lati/'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

res_file = res_dir + 'bayes_iter={:d}_job={:d}.pkl'.format(bayes_iter, job_id)

# create prior distribution
if bayes_iter == 0:
    full_prior = BoxUniform(low =torch.tensor([ 0.0,-0.5,-2.5,-2.0, 0.0],device=device),
                            high=torch.tensor([ 1.0, 1.0,-0.5, 1.0, 0.2],device=device),)
else:
    try:
        with open(f'./../notebooks/phase_ring_lat_posterior_{bayes_iter:d}.pkl','rb') as handle:
            full_prior = pickle.load(handle)
    except:
        with open(f'./../notebooks/phase_ring_lat_posterior.pkl','rb') as handle:
            full_prior = pickle.load(handle)
        
nring = 8

def elong_inp(kl2,gam,ori,phs):
    return 1 + np.cos(phs) * np.exp(-0.5*kl2*(1 + (1-gam**2)/gam**2*np.sin(ori)**2))

def integrate_ring(xea0,xen0,xeg0,xia0,xin0,xig0,inp,Jee,Jei,Jie,Jii,ne,ni,threshe,threshi,nring,
                   dt,Nt,ta=0.01,tn=0.300,tg=0.01,frac_n=0.7):
    '''
    Integrate phase ring with AMPA, NMDA, and GABA receptor dynamics.
    xe0, xi0: initial excitatory and inhibitory activity
    inp: input function, takes time t and returns input at that time
    Jee, Jei, Jie, Jii: connectivity strengths per connection type
    ne, ni: rate activation exponents for excitatory and inhibitory neurons
    threshe, threshi: activation thresholds for excitatory and inhibitory neurons
    nring: number of cells per type in the phase ring
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
    
    dps = np.arange(nring)/nring*2*np.pi
    dps = np.abs(dps[:,None] - dps[None,:])
    dps[dps > np.pi] = 2*np.pi - dps[dps > np.pi]
    
    if np.isscalar(Jee):
        Wee = Jee * np.ones(1)
        Wei = Jei * np.ones(1)
        Wie = Jie * np.ones(1)
        Wii = Jii * np.ones(1)
        
        xea = xea[:,None]
        xen = xen[:,None]
        xeg = xeg[:,None]
        xia = xia[:,None]
        xin = xin[:,None]
        xig = xig[:,None]
        
        nprm = 1
    else:
        Wee = Jee
        Wei = Jei
        Wie = Jie
        Wii = Jii
        
        xea = xea[:,None] * np.ones(len(Jee))[None,:]
        xen = xen[:,None] * np.ones(len(Jee))[None,:]
        xeg = xeg[:,None] * np.ones(len(Jee))[None,:]
        xia = xia[:,None] * np.ones(len(Jee))[None,:]
        xin = xin[:,None] * np.ones(len(Jee))[None,:]
        xig = xig[:,None] * np.ones(len(Jee))[None,:]
        
        nprm = len(Jee)
        
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
        
        net_ee = (Wee*ye.mean(0))[None,:] + ff_inp[:,None]
        net_ei = (Wei*yi.mean(0))[None,:]
        net_ie = (Wie*ye.mean(0))[None,:] + ff_inp[:,None]
        net_ii = (Wii*yi.mean(0))[None,:]
        
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
    def time_event(t,x,ncell,nprm=1):
        int_time = (start_time + max_time) - time.process_time()
        if int_time < 0: int_time = 0
        return int_time
    time_event.terminal = True
    
    sol = integrate.solve_ivp(dyn_func,(0,dt*Nt),y0=x0,t_eval=(Nt*dt,),args=(nring,nprm),method='RK23',events=time_event)
    if sol.status != 0:
        x = np.nan*np.ones(6*nring*nprm)
    else:
        x = sol.y[:,-1]
    x = x.reshape((-1,nprm))
    
    xea = x[0*nring:1*nring,:]
    xen = x[1*nring:2*nring,:]
    xeg = x[2*nring:3*nring,:]
    xia = x[3*nring:4*nring,:]
    xin = x[4*nring:5*nring,:]
    xig = x[5*nring:6*nring,:]
    
    ye = np.fmin(1e5,np.fmax(0,xea+xen+xeg-threshe)**ne)
    yi = np.fmin(1e5,np.fmax(0,xia+xin+xig-threshi)**ni)
    return np.concatenate((xea+xen+xeg,xia+xin+xig)),np.concatenate((ye,yi))

def get_J(theta):
    '''
    theta[:,0] = det(J)/(|Jei| * |Jie|) = 1 - (|Jee| * |Jii|) / (|Jei| * |Jie|)
    theta[:,1] = (Ω_I - Ω_E)/(|Jei| + |Jie|) = 1 - (|Jee| + |Jii|) / (|Jei| + |Jie|)
    theta[:,2] = (log10[|Jei|] + log10[|Jie|]) / 2
    theta[:,3] = (log10[|Jei|] - log10[|Jie|]) / 2
    
    returns: [Jee,Jei,Jie,Jii]
    '''
    Jei = -10**(theta[:,2] + theta[:,3])
    Jie =  10**(theta[:,2] - theta[:,3])
    Jee_p_Jii = (-Jei + Jie) * (1 - theta[:,1])
    Jee_m_Jii_2 = 4*(theta[:,0] * Jei*Jie) + Jee_p_Jii**2
    Jee = 0.5*(Jee_p_Jii + torch.sqrt(Jee_m_Jii_2))
    Jii = -(Jee_p_Jii - Jee)
    
    return Jee,Jei,Jie,Jii

def get_resps(theta, nring=8, gam=0.65):
    Jee,Jei,Jie,Jii = get_J(theta)
    
    c = 100
    nori = 8
    nring = 8
    
    oris = np.linspace(0,np.pi,nori,endpoint=False)
    phss = np.linspace(0,2*np.pi,nring,endpoint=False)
    
    resps = np.zeros((theta.shape[0],2,nori,nring))
    for prm_idx in range(theta.shape[0]):
        print(prm_idx+1,'out of',theta.shape[0])
        thresh = c * theta[prm_idx,4].item()
        for ori_idx,ori in enumerate(oris):
            def ff_inp(t):
                return c*elong_inp(2.0,gam,ori,phss+2*np.pi*3*t)
            _,resp = integrate_ring(np.zeros(nring),np.zeros(nring),np.zeros(nring),
                                np.zeros(nring),np.zeros(nring),np.zeros(nring),
                                ff_inp,Jee[prm_idx].item(),Jei[prm_idx].item(),Jie[prm_idx].item(),Jii[prm_idx].item(),2,2,
                                thresh,thresh,nring,0.25,4*50)
            resps[prm_idx,:,ori_idx,:] = resp.T.reshape(2,nring)
        
    return resps

def simulator(theta):
    '''
    theta[:,0] = det(J)/(|Jei| * |Jie|) = 1 - (|Jee| * |Jii|) / (|Jei| * |Jie|)
    theta[:,1] = (Ω_I - Ω_E)/(|Jei| + |Jie|) = 1 - (|Jee| + |Jii|) / (|Jei| + |Jie|)
    theta[:,2] = (log10[|Jei|] + log10[|Jie|]) / 2
    theta[:,3] = (log10[|Jei|] - log10[|Jie|]) / 2
    theta[:,4] = thresh
    
    returns: [os,mr]
    os = excitatory orientation selectivity
    mr = excitatory modulation ratio
    '''
    
    _,_,_,Jii = get_J(theta)
    
    out = torch.zeros((theta.shape[0],8),dtype=theta.dtype).to(theta.device)
    
    for i,gam in enumerate([0.91,0.78,0.68,0.58]):
        resps = get_resps(theta,gam=gam)
        os,mr = af.calc_OS_MR(resps[:,0,:])
        out[:,0+i] = torch.tensor(os,dtype=theta.dtype).to(theta.device)
        out[:,4+i] = torch.tensor(mr,dtype=theta.dtype).to(theta.device)
    
    valid_idx = torch.all(torch.tensor(resps) < 5e4,axis=(1,2,3)) & (Jii < 0) \
        & torch.tensor((np.mean(resps[:,0,0,:],-1) > np.mean(resps[:,0,nring//2,:],-1)))
    
    return torch.where(valid_idx[:,None],out,torch.tensor([torch.nan])[:,None])

start = time.process_time()

# sample from prior
theta = full_prior.sample((num_samp,))

# simulate sheet
x = simulator(theta)

print(f'Simulating samples took',time.process_time() - start,'s\n')

# save results
with open(res_file, 'wb') as handle:
    pickle.dump({
        'theta': theta,
        'x': x,
    }, handle)
