import numpy as np
from scipy.stats import qmc

def gen_rf_sct_map(N,sig2,sct_scale,pol_scale,EI_match=True,EI_pol_corr=0.65,kern_type='bandpass',seed=0):
    rng = np.random.default_rng(seed)

    ks = np.arange(N)/N
    ks[ks > 0.5] = ks[ks > 0.5] - 1
    kxs,kys = np.meshgrid(ks*N,ks*N)
    ks = np.sqrt(kxs**2 + kys**2)
    kpol = 1/(np.sqrt(sig2)*pol_scale)
    
    if kern_type == 'bandpass':
        k_kern = ks*np.exp(0.125-2*(ks - 0.75*kpol)**2/kpol**2)
    elif kern_type == 'lowpass':
        k_kern = np.exp(-0.5*(ks/kpol)**2)
    elif kern_type == 'highpass':
        k_kern = 1 - np.exp(-0.5*(ks/kpol)**2)

    if EI_match:
        # polmap = (np.fft.ifft2(k_kern*\
        #     np.fft.fft2(rng.binomial(n=1,p=0.5,size=(N,N))-0.5)) > 0).astype(int)
        polmap = (np.fft.ifft2(k_kern*\
            np.fft.fft2(rng.normal(size=(N,N)))) > 0).astype(int)
    
        sctmap = rng.normal(loc=0,scale=np.sqrt(sig2)*sct_scale,size=(N,N,2))
    else:
        # e_field = rng.binomial(n=1,p=0.5,size=(N,N))-0.5
        # i_field = rng.binomial(n=1,p=0.5,size=(N,N))-0.5
        e_field = rng.normal(size=(N,N))
        i_field = rng.normal(size=(N,N))
        s_field = (e_field + i_field) / np.sqrt(2)
        d_field = (e_field - i_field) / np.sqrt(2)
        del e_field, i_field
        polmap_s = (1+EI_pol_corr)*np.fft.ifft2(k_kern*np.fft.fft2(s_field)).real
        polmap_d = (1-EI_pol_corr)*np.fft.ifft2(k_kern*np.fft.fft2(d_field)).real
        polmap_e = (polmap_s + polmap_d) / np.sqrt(2)
        polmap_i = (polmap_s - polmap_d) / np.sqrt(2)
        polmap = ((polmap_e > 0).astype(int), (polmap_i > 0).astype(int))
        
        sctmap = (rng.normal(loc=0,scale=np.sqrt(sig2)*sct_scale,size=(N,N,2)),
                  rng.normal(loc=0,scale=np.sqrt(sig2)*sct_scale,size=(N,N,2)))
    
    return sctmap,polmap

def gen_abs_phs_map(N,rf_sct_map,pol_map,ori,freq,Lgrid,periodic=True):
    xs,ys = np.meshgrid(np.arange(N)/N,np.arange(N)/N)
    abs_rf_centx = rf_sct_map[:,:,0] + xs
    abs_rf_centy = rf_sct_map[:,:,1] + ys
    
    if periodic:
        kx = np.round(freq*Lgrid*np.cos(ori))
        ky = np.round(freq*Lgrid*np.sin(ori))
    else:
        kx = freq*Lgrid*np.cos(ori)
        ky = freq*Lgrid*np.sin(ori)
    
    abs_phs = 2*np.pi*np.mod(kx*abs_rf_centx + ky*abs_rf_centy + 0.5*pol_map,1)
    return abs_phs

# Define function to generate clustered L4 input orientation map
def gen_clst_map(N,dens,bgnd_min,bgnd_max,clst_min,clst_max,meanOS,seed=0,bgnd_scale=4,areaCV=0,bgnd_pow=1,
                 cont_oris=False,cont_sels=False):
    rng = np.random.default_rng(seed)
    
    bgndOS = (bgnd_min+bgnd_pow*bgnd_max)/(bgnd_pow+1)
    clstOS = (clst_min+clst_max)/2

    nclstr = np.round(N**2*dens).astype(int)
    sig2 = (meanOS - bgndOS)/(((clstOS) - bgndOS)*dens*np.pi) / N**2

    rng = np.random.default_rng(seed)

    clstr_pts = qmc.Halton(d=2,scramble=False,seed=seed).random(nclstr)
    
    oris = 2*np.pi*rng.random(nclstr)
    
    if np.isclose(areaCV,0):
        sig2s = sig2*np.ones(nclstr)
        rng.gamma(shape=1,scale=1,size=nclstr)
    else:
        shape = 1/areaCV**2
        scale = sig2/shape
        sig2s = rng.gamma(shape=shape,scale=scale,size=nclstr)
    
    xs,ys = np.meshgrid(np.arange(N)/N,np.arange(N)/N)
    dxs = np.abs(xs[None,:,:] - clstr_pts[:,0,None,None])
    dxs[dxs > 0.5] = 1 - dxs[dxs > 0.5]
    dys = np.abs(ys[None,:,:] - clstr_pts[:,1,None,None])
    dys[dys > 0.5] = 1 - dys[dys > 0.5]
    ds2s = dxs**2 + dys**2

    omap = np.zeros((N,N),dtype='complex64')
    holes = np.zeros((N,N),dtype='float64')
    
    for i in range(nclstr):
        omap += np.heaviside(1.01*sig2s[i]-ds2s[i],1)*np.exp(1j*oris[i])
        holes += np.heaviside(1.01*sig2s[i]-ds2s[i],1)
    holes = np.clip(holes,0,1)
            
    true_clstr_size = np.sum(np.abs(omap))
    omap *= clstOS*nclstr*np.pi*sig2*N**2/true_clstr_size

    ks = np.arange(N)/N
    ks[ks > 0.5] = ks[ks > 0.5] - 1
    kxs,kys = np.meshgrid(ks*N,ks*N)

    bgnd_ofield = np.fft.ifft2(np.exp(-0.5*(kxs**2+kys**2)*sig2*bgnd_scale**2)*\
        np.fft.fft2(np.exp(1j*2*np.pi*rng.random((N,N)))))
    bgnd_ofield /= np.abs(bgnd_ofield)
    bgnd_sfield = bgnd_min+(bgnd_max-bgnd_min)*rng.random((N,N))**bgnd_pow
    clst_sfield = clst_min+(clst_max-clst_min)*rng.random((N,N))
    clst_sfield *= nclstr*np.pi*sig2*N**2/true_clstr_size
    if cont_sels:
        min_clstr_dists = np.min(ds2s,0)
        min_dist = np.min(min_clstr_dists[holes==0])
        max_dist = np.max(min_clstr_dists[holes==0])
        min_clstr_dists[holes==1] = min_dist + (max_dist-min_dist)*rng.random(np.count_nonzero(holes))
        bgnd_sfield = bgnd_sfield.flatten()
        bgnd_sfield[np.argsort(min_clstr_dists.flatten())[::-1]] = np.sort(bgnd_sfield).flatten()
        bgnd_sfield = bgnd_sfield.reshape(N,N)
        
        min_clstr_dists = np.min(ds2s,0)
        min_dist = np.min(min_clstr_dists[holes==1])
        max_dist = np.max(min_clstr_dists[holes==1])
        clst_sfield = clst_sfield.flatten()
        min_clstr_dists[holes==0] = min_dist + (max_dist-min_dist)*rng.random(np.count_nonzero(1-holes))
        clst_sfield[np.argsort(min_clstr_dists.flatten())[::-1]] = np.sort(clst_sfield).flatten()
        clst_sfield = clst_sfield.reshape(N,N)
    if cont_oris:
        omap = (bgnd_sfield*(1-holes)+clst_sfield*holes)*bgnd_ofield
    else:
        omap += bgnd_sfield*bgnd_ofield*(1-holes)
    
    return omap