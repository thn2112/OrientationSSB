import sys
import os
sys.path.insert(0, './..')

from functools import cache
import pickle
from math import floor, ceil
import numpy as np
from scipy.interpolate import griddata,interp1d
from scipy.integrate import quad
from scipy.optimize import curve_fit
from scipy.special import erf
import matplotlib.pyplot as plt
from scipy.stats import rankdata

def circ_corrcoef(x,y,P=2*np.pi):
    assert len(x) == len(y), "Input arrays must have the same length."
    xbar = np.angle(np.mean(np.exp(1j*x*2*np.pi/P)))
    ybar = np.angle(np.mean(np.exp(1j*y*2*np.pi/P)))
    sin_corrcoef = np.mean(np.sin(x - xbar) * np.sin(y - ybar)) / np.sqrt(np.mean(np.sin(x - xbar)**2) * np.mean(np.sin(y - ybar)**2))
    cos_corrcoef = np.mean(np.cos(x - xbar) * np.cos(y - ybar)) / np.sqrt(np.mean(np.cos(x - xbar)**2) * np.mean(np.cos(y - ybar)**2))
    return sin_corrcoef,cos_corrcoef

def mardia_rho(x,y):
    assert len(x) == len(y), "Input arrays must have the same length."
    if not (np.isnan(x) == np.isnan(y)).all():
        nan_idxs = np.isnan(x) | np.isnan(y)
        xcopy = x[~nan_idxs]
        ycopy = y[~nan_idxs]
    else:
        xcopy = x
        ycopy = y
    
    xrank = rankdata(xcopy)
    yrank = rankdata(ycopy)
    n = len(xrank)
    r1 = (np.sum(np.cos(2*np.pi*(xrank-yrank)/n))**2 + np.sum(np.sin(2*np.pi*(xrank-yrank)/n))**2) / n**2
    r2 = (np.sum(np.cos(2*np.pi*(xrank+yrank)/n))**2 + np.sum(np.sin(2*np.pi*(xrank+yrank)/n))**2) / n**2
    return np.fmax(r1,r2)

@cache
def get_bin_cents(Nax,nbins):
    max_val = np.ceil(Nax//2*np.sqrt(2))
    return np.linspace(0,max_val,nbins+1)

@cache
def get_s_idxs(Nax,nbins,fftshift=True):
    max_val = np.ceil(Nax//2*np.sqrt(2))
    step = max_val / nbins
    if fftshift:
        x,y = np.meshgrid(np.arange(-Nax//2,Nax//2),np.arange(-Nax//2,Nax//2))
    else:
        x,y = np.meshgrid(np.arange(Nax),np.arange(Nax))
        x[x > Nax//2] = Nax - x[x > Nax//2]
        y[y > Nax//2] = Nax - y[y > Nax//2]
    return np.digitize(np.sqrt(x**2+y**2),np.linspace(0,max_val,nbins+1)+step/2)

@cache
def get_ds_idxs(Nax,nbins):
    max_val = np.ceil(Nax//2*np.sqrt(2))
    step = max_val / nbins
    x,y = np.meshgrid(np.arange(Nax),np.arange(Nax))
    dx = np.abs(x[:,:,None,None] - x[None,None,:,:])
    dy = np.abs(y[:,:,None,None] - y[None,None,:,:])
    dx[dx > Nax//2] = Nax - dx[dx > Nax//2]
    dy[dy > Nax//2] = Nax - dy[dy > Nax//2]
    return np.digitize(np.sqrt(dx**2 + dy**2).reshape(Nax**2,Nax**2),np.linspace(0,max_val,nbins+1)+step/2)

def get_fps(A,grid_axes=None,nbins=None,zero_mean=True,calc_err=False):
    grid_axes = (-2,-1) if grid_axes is None else grid_axes
    assert len(grid_axes) == 2, "Grid axes must be length 2."
    Nax = A.shape[grid_axes[0]]
    
    # shift grid axes to end and check they are square
    Amove = np.moveaxis(A,grid_axes,(-2,-1))
    assert Amove.shape[-2] == Amove.shape[-1], "Input array must have equal length in given axes."
    
    if nbins is None:
        nbins = int(np.round(np.ceil(Nax//2*np.sqrt(2))))
    if zero_mean:
        fps = np.abs(np.fft.fftshift(np.fft.fft2(Amove - np.nanmean(Amove,axis=(-2,-1),keepdims=True))))**2
    else:
        fps = np.abs(np.fft.fftshift(np.fft.fft2(A)))**2
    raps = np.zeros(Amove.shape[:-2] + (nbins,))
    if calc_err:
        raps_err = np.zeros(Amove.shape[:-2] + (nbins,))

    bin_idxs = get_s_idxs(Nax,nbins)
    for idx in range(0,nbins):
        raps[...,idx] = np.mean(fps[...,bin_idxs == idx],-1)
        if calc_err:
            raps_err[...,idx] = np.std(fps[...,bin_idxs == idx],-1) / np.sqrt(np.sum(...,bin_idxs == idx),-1)
    
    if calc_err:
        return fps,raps,raps_err
    else:
        return fps,raps
    
def get_corr(A,grid_axes=None,patt_axes=None,nbins=None,calc_err=False):
    patt_axes = (-1,) if patt_axes is None else patt_axes
    grid_axes = (-len(patt_axes)-1,) if grid_axes is None else grid_axes
    if np.isscalar(grid_axes):
        grid_axes = (grid_axes,)
    assert len(grid_axes) <= 2, "Grid axes must be length 2 or less."
    Ngrid = 1
    for ax in grid_axes:
        Ngrid *= A.shape[ax]
    Nax = int(np.round(np.sqrt(Ngrid)))
    
    # shift grid and pattern axes to end and check grid axes are square
    Amove = np.moveaxis(A,grid_axes + patt_axes,tuple(range(-(len(grid_axes)+len(patt_axes)),0)))
    assert Nax**2 == Ngrid, "Grid dimensions must be a perfect square."
    
    if nbins is None:
        nbins = int(np.round(np.ceil(Nax//2*np.sqrt(2))))

    Az = Amove.reshape(Amove.shape[:-len(grid_axes)-len(patt_axes)] + (Nax**2,-1))
    Npatt = Az.shape[-1]
    Az = Az - np.mean(Az,axis=-1,keepdims=True)
    Az = Az / np.std(Az,axis=-1,keepdims=True)
    
    corr = np.zeros(Az.shape[:-2] + (Nax**2,Nax**2))
    for i in range(Npatt):
        corr += Az[...,None,:,i] * Az[...,:,None,i]
    corr /= Npatt
    corr_curve = np.zeros(Az.shape[:-2] + (nbins,))
    if calc_err:
        corr_err = np.zeros(Az.shape[:-2] + (nbins,))
    
    bin_idxs = get_ds_idxs(Nax,nbins)
    for idx in range(nbins):
        corr_curve[...,idx] = np.mean(corr[...,bin_idxs == idx],-1)
        if calc_err:
            corr_err[...,idx] = np.std(corr[...,bin_idxs == idx],-1) / np.sqrt(np.sum(...,bin_idxs == idx),-1)
    
    if calc_err:
        return corr,corr_curve,corr_err
    else:
        return corr,corr_curve

def get_ori_sel(opm,calc_fft=True):
    N4 = opm.shape[0]
    sel = np.abs(opm)
    ori = np.angle(opm)/2
    ori = ori - (np.sign(ori)-1)*0.5*np.pi
    ori *= 180/np.pi
    
    if calc_fft:
        opm_fft,opm_fps = get_fps(opm)
        return ori,sel,opm_fft,opm_fps
    else:
        return ori,sel
    
def calc_hypercol_size(fps,N):
    def fps_fn(k,a0,a1,a2,a3,a4,a5):
        return a0*np.exp(-0.5*(k-a1)**2/a2**2) + a3 + a4*k + a5*k**2

    freqs = np.arange(N//2)/N
    peak_idx = np.argmax(np.concatenate(([0,0],fps[2:N//2])))

    popt,pcov = curve_fit(fps_fn,freqs,fps[:N//2],
                          p0=(fps[peak_idx],freqs[peak_idx],0.5*freqs[peak_idx],0,0,0))
    
    hcsize = 1/popt[1]
    return hcsize,fps_fn(np.arange(len(fps))/N,*popt)

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def cross(A,B):
    return A[0]*B[1] - A[1]*B[0]

def intersectpt(A,B,C,D):
    qmp = [C[0]-A[0],C[1]-A[1]]
    r = [B[0]-A[0],B[1]-A[1]]
    s = [D[0]-C[0],D[1]-C[1]]
    rxs = cross(r,s)
    
    t = cross(qmp,s)/rxs
#     u = cross(qmp,r)/rxs
    
    return [A[0]+t*r[0],A[1]+t*r[1]]

def calc_pinwheels(A):
    rcont = plt.contour(np.real(A),levels=[0],colors="C0")
    icont = plt.contour(np.imag(A),levels=[0],colors="C1")

    rsegpts = []
    for pts in rcont.allsegs[0]:
        for i in range(len(pts)-1):
            rsegpts.append([pts[i],pts[i+1]])
    rsegpts = np.array(rsegpts)

    isegpts = []
    for pts in icont.allsegs[0]:
        for i in range(len(pts)-1):
            isegpts.append([pts[i],pts[i+1]])
    isegpts = np.array(isegpts)
    
    pwcnt = 0
    pwpts = []

    for rsegpt in rsegpts:
        for isegpt in isegpts:
            if intersect(rsegpt[0],rsegpt[1],isegpt[0],isegpt[1]):
                pwcnt += 1
                pwpts.append(intersectpt(rsegpt[0],rsegpt[1],isegpt[0],isegpt[1]))
    pwpts = np.array(pwpts)
    
    return pwcnt,pwpts

def raps_fn(k,alf,a0,a1,a2,a3):
    return a0*k*np.exp(-0.5*(k-a1)**2/a2**a3) * 0.5*(1+erf(alf*(k-a1)/(np.sqrt(2)*a2)))

def fit_raps_fn(freqs,raps):
    peak_idx = np.argmax(np.concatenate(([0,0],raps[2:])))
    popt,pcov = curve_fit(raps_fn,freqs,raps,
                        p0=(0,raps[peak_idx]/freqs[peak_idx],freqs[peak_idx],0.3*freqs[peak_idx],2),
                        bounds=([-100,0,0,-np.inf,1],[100,np.inf,np.inf,np.inf,4]),
                        maxfev=100000)
    raps_itp = lambda k: raps_fn(k/(2*np.pi),*popt)
    
    return raps_itp, popt, pcov

def calc_pinwheel_density_from_raps_func(raps_func,lim=np.inf):
    norm = quad(raps_func,0,lim)[0]
    return quad(lambda k: k**3*raps_func(k)/norm,0,lim)[0] \
        / quad(lambda k: k*raps_func(k)/norm,0,lim)[0]**3 * np.pi

def calc_pinwheel_density_from_raps(freqs,raps,continuous=True,return_fit=False):        
    if continuous:
        # raps_itp = interp1d(freqs,raps,kind='linear',bounds_error=False,fill_value=0)
        if raps.ndim == 1:
            try:
                raps_itp,popt,_ = fit_raps_fn(freqs,raps)
            except:
                if return_fit:
                    return np.nan, [np.nan]*5
                else:
                    return np.nan
            
            pwd = calc_pinwheel_density_from_raps_func(raps_itp)
            if return_fit:
                return pwd, popt
            else:
                return pwd
        else:
            pwd_list = []
            popt_list = []
            if freqs.ndim == 1:
                freqs = freqs[None,:] * np.ones((raps.shape[0],1))
            elif freqs.shape[0] != raps.shape[0]:
                freqs = freqs * np.ones((raps.shape[0],1))
            for i in range(raps.shape[0]):
                try:
                    raps_itp,popt,_ = fit_raps_fn(freqs[i],raps[i])
                except:
                    pwd_list.append(np.nan)
                    if return_fit:
                        popt_list.append([np.nan]*5)
                    continue
                
                pwd_list.append(calc_pinwheel_density_from_raps_func(raps_itp))
                if return_fit:
                    popt_list.append(popt)
            if return_fit:
                return np.array(pwd_list), np.array(popt_list)
            else:
                return np.array(pwd_list)
    else:
        norm = np.sum(raps,-1,keepdims=True)
        return np.sum(freqs**3 * raps / norm,-1) \
            / np.sum(freqs * raps / norm,-1)**3 * np.pi

def bandpass_filter(A,ll,lu):
    Nax = A.shape[0]
    
    xs,ys = np.meshgrid(np.arange(Nax)/Nax,np.arange(Nax)/Nax)
    xs[xs > 0.5] = 1 - xs[xs > 0.5]
    ys[ys > 0.5] = 1 - ys[ys > 0.5]
    ks = 2*np.pi*np.sqrt(xs**2 + ys**2)

    def fermi_kernel(lamlp,beta=0.05):
        return 1 / (1 + np.exp(-(2*np.pi/lamlp - ks)/beta))
    
    A_fft = np.fft.fft2(A)
    
    this_kern = fermi_kernel(ll) * (1-fermi_kernel(lu))
    return np.fft.ifft2(A_fft*this_kern)
    
def calc_dc_ac_comp(A,axis=-1):
    if A.ndim > 1:
        A_xpsd = np.moveaxis(A,axis,-1)
    else:
        A_xpsd = A.copy()
    Nax = A.shape[axis]
    angs = np.arange(Nax) * 2*np.pi/Nax
    A0 = np.mean(A_xpsd,axis=-1)
    As = np.mean(A_xpsd*np.sin(angs),axis=-1)
    Ac = np.mean(A_xpsd*np.cos(angs),axis=-1)
    A1mod = np.sqrt(As**2+Ac**2)
    A1phs = np.arctan2(As,Ac)
    
    return A0,A1mod,A1phs

def calc_OPM(A):
    # calculate orientation DC, AC, and center from phase-averaged response
    A0,A1,PO = calc_dc_ac_comp(A)
    
    # calculate OS from DC and AC of phase-averaged response
    OS = np.where(A1==0,0,A1/A0)
    
    return OS * np.exp(1j*PO)

def calc_OPM_MR(A):
    noris = np.shape(A)[-2]
    
    # calculate phase DC and AC per orientation
    F0,F1,_ = calc_dc_ac_comp(A)
    
    # calculate orientation DC, AC, and center from phase-averaged response
    A0,A1,PO = calc_dc_ac_comp(F0)
    
    # calculate OS from DC and AC of phase-averaged response
    OPM = np.where(A1==0,0,A1/A0) * np.exp(1j*PO)
    
    # infer index of preferred orientation
    PO += np.pi/noris
    PO = np.mod(PO,2*np.pi)
    PO = np.array(PO / (2*np.pi) * noris).astype(int)
    PO = np.mod(PO,noris)
    
    # calculate MR at preferred orientation
    # pref_F0,pref_F1 = np.zeros(np.shape(A)[:-2]),np.zeros(np.shape(A)[:-2])
    F0,F1 = np.take_along_axis(F0,PO[...,None],-1)[...,0],np.take_along_axis(F1,PO[...,None],-1)[...,0]
    MR = np.where(F1==0,0,2*F1/F0)
    
    return OPM,MR

def calc_OS_MR(A):
    noris = np.shape(A)[-2]
    
    # calculate phase DC and AC per orientation
    F0,F1,_ = calc_dc_ac_comp(A)
    
    # calculate orientation DC, AC, and center from phase-averaged response
    A0,A1,PO = calc_dc_ac_comp(F0)
    
    # calculate OS from DC and AC of phase-averaged response
    OS = np.where(A1==0,0,A1/A0)
    
    # infer index of preferred orientation
    PO += np.pi/noris
    PO = np.mod(PO,2*np.pi)
    PO = np.array(PO / (2*np.pi) * noris).astype(int)
    PO = np.mod(PO,noris)
    
    # calculate MR at preferred orientation
    # pref_F0,pref_F1 = np.zeros(np.shape(A)[:-2]),np.zeros(np.shape(A)[:-2])
    F0,F1 = np.take_along_axis(F0,PO[...,None],-1)[...,0],np.take_along_axis(F1,PO[...,None],-1)[...,0]
    MR = np.where(F1==0,0,2*F1/F0)
    
    return OS,MR

def layout_w(w,ngrid,rarb,nskip=0,pre=True):
    # calculate arbor diameter
    darb = 2*rarb+1
    
    # layout pre/postsynaptic weight of each shown cell in subsquares in larger grid
    nshow = ngrid//(1+nskip)
    w_laid = np.zeros((nshow*(darb+1)+1,nshow*(darb+1)+1))
    for i in range(nshow):
        for j in range(nshow):
            if pre: # extract presynaptic weights for cell at (i,j)
                this_w = w.reshape(ngrid,ngrid,ngrid,ngrid)[i*(1+nskip),j*(1+nskip),:,:]
            else: # extract postsynaptic weights for cell at (i,j)
                this_w = w.reshape(ngrid,ngrid,ngrid,ngrid)[:,:,i*(1+nskip),j*(1+nskip)]
            
            # place weights grid subsquare
            w_laid[1+i*(darb+1):1+i*(darb+1)+darb,1+j*(darb+1):1+j*(darb+1)+darb] =\
                np.roll(this_w,(rarb-i*(1+nskip),rarb-j*(1+nskip)),axis=(-2,-1))[:darb,:darb]
    return w_laid

def get_rf_fft_resps(rfs,ngrid,noris):
    xs,ys = np.meshgrid(np.arange(1,ngrid+1)/ngrid - 0.5,np.arange(1,ngrid+1)/ngrid - 0.5)
    xs = np.roll(xs,(ngrid//2+1,ngrid//2+1),axis=(0,1))
    ys = np.roll(ys,(ngrid//2+1,ngrid//2+1),axis=(0,1))
    
    ang_bins = np.zeros((ngrid,ngrid),dtype=int)
    ang_bins = np.digitize(np.arctan2(xs,ys),np.linspace(-0.5*np.pi/noris,np.pi-0.5*np.pi/noris,noris+1))
    ang_bins[0,0] = 0
    ang_bins[np.sqrt(xs**2 + ys**2) >= 0.5] = 0
    
    rf_ffts = np.abs(np.fft.fft2(rfs))
    rf_fft_angs = np.zeros((ngrid,ngrid,noris))
    
    for idx in range(noris):
        rf_fft_angs[:,:,idx] = np.max(rf_ffts[:,:,ang_bins==idx+1],-1)
        
    return rf_fft_angs

def scat_map_to_grid(scat_map,xs,ys,ngrid,per_pad=None,indexing='xy',method='linear'):
    # Given map on uneven grid, interpolate to regular grid
    # scat_map, xs, ys must all be 2D arrays of the same shape
    if per_pad is None:
        per_pad = int(np.round(ngrid/4))
        
    scat_map_extended = np.block([
        [scat_map[-per_pad:,-per_pad:],scat_map[-per_pad:,:],scat_map[-per_pad:,:per_pad]],
        [scat_map[:,-per_pad:],scat_map,scat_map[:,:per_pad]],
        [scat_map[:per_pad,-per_pad:],scat_map[:per_pad,:],scat_map[:per_pad,:per_pad]]
    ])
    
    if indexing == 'xy':
        xs_extended = np.block([
            [xs[-per_pad:,-per_pad:]-1,xs[-per_pad:,:],xs[-per_pad:,:per_pad]+1],
            [xs[:,-per_pad:]-1,xs,xs[:,:per_pad]+1],
            [xs[:per_pad,-per_pad:]-1,xs[:per_pad,:],xs[:per_pad,:per_pad]+1]
        ])
        ys_extended = np.block([
            [ys[-per_pad:,-per_pad:]-1,ys[-per_pad:,:]-1,ys[-per_pad:,:per_pad]-1],
            [ys[:,-per_pad:],ys,ys[:,:per_pad]],
            [ys[:per_pad,-per_pad:]+1,ys[:per_pad,:]+1,ys[:per_pad,:per_pad]+1]
        ])
    elif indexing == 'ij':
        xs_extended = np.block([
            [xs[-per_pad:,-per_pad:]-1,xs[-per_pad:,:]-1,xs[-per_pad:,:per_pad]-1],
            [xs[:,-per_pad:],xs,xs[:,:per_pad]],
            [xs[:per_pad,-per_pad:]+1,xs[:per_pad,:]+1,xs[:per_pad,:per_pad]+1]
        ])
        ys_extended = np.block([
            [ys[-per_pad:,-per_pad:]-1,ys[-per_pad:,:],ys[-per_pad:,:per_pad]+1],
            [ys[:,-per_pad:]-1,ys,ys[:,:per_pad]+1],
            [ys[:per_pad,-per_pad:]-1,ys[:per_pad,:],ys[:per_pad,:per_pad]+1]
        ])
    else:
        raise ValueError("indexing must be 'xy' or 'ij'")
    
    # interpolate to grid
    grid_xs,grid_ys = np.meshgrid(np.linspace(0.5/ngrid,1-0.5/ngrid,ngrid),
                                  np.linspace(0.5/ngrid,1-0.5/ngrid,ngrid),indexing=indexing)
    grid_map = griddata((xs_extended.flatten(),ys_extended.flatten()),
                        scat_map_extended.flatten(),(grid_xs,grid_ys),method=method)
    
    return grid_map