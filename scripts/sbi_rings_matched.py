import os
import pickle
import time
import argparse

from itertools import product

import numpy as np
import torch
from scipy import integrate
from scipy import interpolate
from scipy.linalg import circulant

from sbi.utils import BoxUniform
from sbi.utils.user_input_checks import process_prior

import analyze_func as af
import map_func as mf

parser = argparse.ArgumentParser()
parser.add_argument('--job_id', '-i', help='completely arbitrary job id label',type=int, default=0)
parser.add_argument('--num_samp', '-ns', help='number of samples',type=int, default=200)
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

res_dir = res_dir + 'sbi_rings/'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

res_file = res_dir + 'bayes_iter={:d}_job={:d}.pkl'.format(bayes_iter, job_id)

# create prior distribution
if bayes_iter == 0:
    '''
    theta[:,0] = det(J)/(|Jei| * |Jie|) = 1 - (|Jee| * |Jii|) / (|Jei| * |Jie|)
    theta[:,1] = (Ω_I - Ω_E)/(|Jei| + |Jie|) = 1 - (|Jee| + |Jii|) / (|Jei| + |Jie|)
    theta[:,2] = (log10[|Jei|] + log10[|Jie|]) / 2
    theta[:,3] = (log10[|Jei|] - log10[|Jie|]) / 2
    theta[:,4] = s_n
    theta[:,5] = s_b
    theta[:,6] = log2(Je_broad / Je_narrow)
    theta[:,7] = log2(Ji_broad / Ji_narrow)
    '''
    full_prior = BoxUniform(
        low =torch.tensor([0.0,-0.5,-2.5,-2.0, 0.2, 1.0,-2.0,-2.0],device=device),
        high=torch.tensor([1.0, 1.0,-0.5, 1.0, 1.0, 4.0, 1.0, 1.0],device=device),)
else:
    try:
        with open(f'./../notebooks/rings_posterior_{bayes_iter:d}.pkl','rb') as handle:
            full_prior = pickle.load(handle)
    except:
        with open(f'./../notebooks/rings_posterior.pkl','rb') as handle:
            full_prior = pickle.load(handle)

N = 60

nos = 4
nori = 8
nphs = 8

nperpop = nos*nori*nphs

kl2 = 2

def elong_inp(gam,ori,phs):
    return 1 + np.cos(phs)*np.exp(-kl2*(1+(1-gam**2)/gam**2*np.sin(ori)**2)/2)

gams = np.linspace(0.4,1,301)
resps = np.fmax(0,elong_inp(gams[:,None,None],np.linspace(0,np.pi,36,endpoint=False)[None,:,None],np.linspace(0,2*np.pi,36,endpoint=False)[None,None,:])-1)**2
oss,_ = af.calc_OS_MR(resps)

gam_os_itp = interpolate.interp1d(oss,gams,fill_value='extrapolate')
gams = (gam_os_itp(np.array([0.1,0.3,0.5,0.7]))[:,None,None] * np.ones((1,nori,nphs))).flatten()

def bin_diff(diff,nring):
    half_nring = nring//2
    diff_bins = np.digitize(diff.flatten()/np.pi,bins=np.arange(half_nring)/half_nring+0.5/half_nring).reshape(diff.shape)
    return diff_bins

# create L4 orientation map
rng = np.random.default_rng(0)
opm_fft = rng.normal(size=(N,N)) + 1j * rng.normal(size=(N,N))
opm_fft[0,0] = 0

freqs = np.fft.fftfreq(N,1/N)
freqs = np.sqrt(freqs[:,None]**2 + freqs[None,:]**2)

decay = 5
opm_fft *= np.exp(-freqs/decay)

opm = np.fft.ifft2(opm_fft)

opm *= np.abs(opm)**1.6/np.abs(opm)
opm *= 0.16 / np.median(np.abs(opm)) # normalize median to data
opm *= np.clip(np.abs(opm),0,0.8) / np.abs(opm) # clip max os to 0.8

opms = np.concatenate((opm.flatten(),opm.flatten()))
pos = np.angle(opms)
oss = np.abs(opms)
dpos = np.abs(pos[:,None] - pos[None,:])
dpos[dpos > np.pi] = 2*np.pi - dpos[dpos > np.pi]
dpos = dpos.reshape((2,N**2,2,N**2))
dpos_bins = bin_diff(dpos,nori)
os_bins = np.digitize(oss.flatten(),bins=np.array([0.2,0.4,0.6]),right=True).reshape((2,N**2))

# create L4 phase map
sig2 = 0.00095

rf_sct_scale = 0.8
pol_scale = 2.0
L_mm = N/11
mag_fact = 0.02
L_deg = L_mm / np.sqrt(mag_fact)
grate_freq = 0.06

rf_sct_map,pol_map = mf.gen_rf_sct_map(N,sig2,rf_sct_scale,pol_scale,EI_match=True)
abs_phs = mf.gen_abs_phs_map(N,rf_sct_map,pol_map,0,grate_freq,L_deg)

abs_phss = np.concatenate((abs_phs.flatten(),abs_phs.flatten()))
dphss = np.abs(abs_phss[:,None] - abs_phss[None,:])
dphss[dphss > np.pi] = 2*np.pi - dphss[dphss > np.pi]
dphss = dphss.reshape((2,N**2,2,N**2))

dphss_bins = bin_diff(dphss,nphs)

xs,ys = np.meshgrid(np.arange(N)/N,np.arange(N)/N)
dxs = np.abs(xs[:,:,None,None] - xs[None,None,:,:])
dxs[dxs > 0.5] = 1 - dxs[dxs > 0.5]
dys = np.abs(ys[:,:,None,None] - ys[None,None,:,:])
dys[dys > 0.5] = 1 - dys[dys > 0.5]
ds2s = (dxs**2 + dys**2).reshape(N**2,N**2)

def calc_w_binned(w_full):
    w_binned = np.zeros((2,nos,nori,nphs,2,nos,nori,nphs))
    
    for (prepop_idx, preos_idx) in product(range(2), range(nos)):
        pre_idx = os_bins[prepop_idx] == preos_idx
        for (postpop_idx, postos_idx) in product(range(2), range(nos)):
            post_idx = os_bins[postpop_idx] == postos_idx
            for dpo_idx in range(nori//2+1):
                dpo_circ_vec = np.zeros(nori)
                dpo_circ_vec[dpo_idx] = 1
                dpo_circ_vec[(nori-dpo_idx)%nori] = 1
                for dphs_idx in range(nphs//2+1):
                    dphs_circ_vec = np.zeros(nphs)
                    dphs_circ_vec[dphs_idx] = 1
                    dphs_circ_vec[(nphs-dphs_idx)%nphs] = 1
                    
                    idxs = pre_idx[None,:] & post_idx[:,None] &\
                        (dpos_bins[postpop_idx,:,prepop_idx,:] == dpo_idx) &\
                        (dphss_bins[postpop_idx,:,prepop_idx,:] == dphs_idx)
                    this_w_binned = w_full[postpop_idx,:,prepop_idx,:][idxs].mean()
                    this_w_binned *= np.count_nonzero(idxs,-1)[np.any(idxs,-1)].mean()
                    w_binned[postpop_idx,postos_idx,:,:,prepop_idx,preos_idx,:,:] +=\
                        this_w_binned * circulant(dpo_circ_vec)[:,None,:,None] * circulant(dphs_circ_vec)[None,:,None,:]
    
    return w_binned

def integrate_ring(xea0,xen0,xeg0,xia0,xin0,xig0,inp,Jee,Jei,Jie,Jii,ne,ni,threshe,threshi,
                   dt,Nt,ta=0.01,tn=0.300,tg=0.01,frac_n=0.7,s_n=0.5,s_b=2,
                   frac_e_broad=0.5,frac_i_broad=0.5):
    '''
    Integrate phase ring with AMPA, NMDA, and GABA receptor dynamics.
    xe0, xi0: initial excitatory and inhibitory activity
    inp: input function, takes time t and returns input at that time
    Jee, Jei, Jie, Jii: connectivity strengths per connection type
    ne, ni: rate activation exponents for excitatory and inhibitory neurons
    threshe, threshi: activation thresholds for excitatory and inhibitory neurons
    dt: time step for integration
    Nt: number of time steps to integrate
    ta, tn, tg: time constants for AMPA, NMDA, and GABA receptor dynamics
    frac_n: fraction of NMDA vs NMDA+AMPA receptors in the excitatory popubroad_narrowon
    '''
    
    xea = xea0.copy()
    xen = xen0.copy()
    xeg = xeg0.copy()
    xia = xia0.copy()
    xin = xin0.copy()
    xig = xig0.copy()
    
    if np.isscalar(Jee):
        kernn = np.exp(-ds2s/s_n**2)
        norm = np.sum(kernn,axis=-1,keepdims=True)
        kernn /= norm
        kernb = np.exp(-ds2s/s_b**2)
        norm = np.sum(kernb,axis=-1,keepdims=True)
        kernb /= norm
        w_full = np.zeros((2,N**2,2,N**2))
        w_full[0,:,0,:] = Jee * (kernn + frac_e_broad * kernb)
        w_full[0,:,1,:] = Jei * (kernn + frac_i_broad * kernb)
        w_full[1,:,0,:] = Jie * (kernn + frac_e_broad * kernb)
        w_full[1,:,1,:] = Jii * (kernn + frac_i_broad * kernb)
        
        w_binned = calc_w_binned(w_full)
        Wee = w_binned[0,:,:,:,0,:,:,:].reshape(nperpop,nperpop)
        Wei = w_binned[0,:,:,:,1,:,:,:].reshape(nperpop,nperpop)
        Wie = w_binned[1,:,:,:,0,:,:,:].reshape(nperpop,nperpop)
        Wii = w_binned[1,:,:,:,1,:,:,:].reshape(nperpop,nperpop)
        del kernn, kernb, norm, w_full, w_binned

        Wee = Wee[:,:,None]
        Wei = Wei[:,:,None]
        Wie = Wie[:,:,None]
        Wii = Wii[:,:,None]
        
        xea = xea[:,None]
        xen = xen[:,None]
        xeg = xeg[:,None]
        xia = xia[:,None]
        xin = xin[:,None]
        xig = xig[:,None]
        
        nprm = 1
    else:
        raise NotImplementedError("Only single batched parameters are implemented.")
        
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
    def time_event(t,x,ncell,nprm=1):
        int_time = (start_time + max_time) - time.process_time()
        if int_time < 0: int_time = 0
        return int_time
    time_event.terminal = True
    
    sol = integrate.solve_ivp(dyn_func,(0,dt*Nt),y0=x0,t_eval=(Nt*dt,),args=(nperpop,nprm),method='RK23',events=time_event)
    if sol.status != 0:
        x = np.nan*np.ones(6*nperpop*nprm)
    else:
        x = sol.y[:,-1]
    x = x.reshape((-1,nprm))
    
    xea = x[0*nperpop:1*nperpop,:]
    xen = x[1*nperpop:2*nperpop,:]
    xeg = x[2*nperpop:3*nperpop,:]
    xia = x[3*nperpop:4*nperpop,:]
    xin = x[4*nperpop:5*nperpop,:]
    xig = x[5*nperpop:6*nperpop,:]
    
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

def get_resps(theta):
    Jee,Jei,Jie,Jii = get_J(theta)
    
    c = 100
    thresh = c
    
    oris = (np.linspace(0,np.pi,nori,endpoint=False)[None,:,None] * np.ones((nos,1,nphs))).flatten()
    phss = (np.linspace(0,2*np.pi,nphs,endpoint=False)[None,None,:] * np.ones((nos,nori,1))).flatten()
    
    resps = np.zeros((theta.shape[0],2,nos,nori,nphs))
    for prm_idx in range(theta.shape[0]):
        def ff_inp(t):
            return c*elong_inp(gams,oris,phss+2*np.pi*3*t)
        _,resp = integrate_ring(np.zeros(nperpop),np.zeros(nperpop),np.zeros(nperpop),
                            np.zeros(nperpop),np.zeros(nperpop),np.zeros(nperpop),
                            ff_inp,Jee[prm_idx].item(),Jei[prm_idx].item(),Jie[prm_idx].item(),Jii[prm_idx].item(),2,2,
                            thresh,thresh,0.25,4*50,
                            s_n=theta[prm_idx,4].item() * np.sqrt(sig2),s_b=theta[prm_idx,5].item() * np.sqrt(sig2),
                            frac_e_broad=2**theta[prm_idx,6].item(),frac_i_broad=2**theta[prm_idx,7].item())
        resps[prm_idx,:,:,:,:] = resp.reshape(2,nos,nori,nphs)
        
    return resps

def simulator(theta):
    '''
    theta[:,0] = det(J)/(|Jei| * |Jie|) = 1 - (|Jee| * |Jii|) / (|Jei| * |Jie|)
    theta[:,1] = (Ω_I - Ω_E)/(|Jei| + |Jie|) = 1 - (|Jee| + |Jii|) / (|Jei| + |Jie|)
    theta[:,2] = (log10[|Jei|] + log10[|Jie|]) / 2
    theta[:,3] = (log10[|Jei|] - log10[|Jie|]) / 2
    theta[:,4] = s_n
    theta[:,5] = s_b
    theta[:,6] = log2(Je_broad / Je_narrow)
    theta[:,7] = log2(Ji_broad / Ji_narrow)
    
    returns: [os,mr]
    os = excitatory orientation selectivity
    mr = excitatory modulation ratio
    '''
    
    _,_,_,Jii = get_J(theta)
    
    out = torch.zeros((theta.shape[0],8),dtype=theta.dtype).to(theta.device)
    
    resps = get_resps(theta)
    os,mr = af.calc_OS_MR(resps[:,0])
    out[:,:4] = torch.tensor(os,dtype=theta.dtype).to(theta.device)
    out[:,4:] = torch.tensor(mr,dtype=theta.dtype).to(theta.device)
    
    valid_idx = torch.all(torch.tensor(resps) < 5e4,axis=(1,2,3,4)) & (Jii < 0) \
        & torch.tensor((np.mean(resps[:,0,-1,0,:],-1) > np.mean(resps[:,0,-1,nphs//2,:],-1)))
    
    return torch.where(valid_idx[:,None],out,torch.tensor([torch.nan])[:,None])

thetas = torch.zeros((0,8))
xs = torch.zeros((0,8))

while thetas.shape[0] < num_samp:
    this_samps = min(3, num_samp - thetas.shape[0])
    
    start = time.process_time()
    # sample from prior
    theta = full_prior.sample((this_samps,))
    # simulate sheet
    x = simulator(theta)

    thetas = torch.cat([thetas,theta],dim=0)
    xs = torch.cat([xs,x],dim=0)

    print(f'Simulating samples took',time.process_time() - start,'s\n')

    # save results
    with open(res_file, 'wb') as handle:
        pickle.dump({
            'theta': thetas,
            'x': xs,
        }, handle)
