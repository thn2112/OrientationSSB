import numpy as np

def elong_het_transf(shape,mean_ecc=0.7,std_ecc=0.0175,std_size=0.0021,dxs=None,dys=None,N=None,seed=0):
    if dxs is None or dys is None:
        assert N is not None
        xs,ys = np.meshgrid(np.arange(N)/N,np.arange(N)/N)
        dxs = xs[:,:,None,None] - xs[None,None,:,:]
        dxs[dxs > 0.5] =  dxs[dxs > 0.5] - 1
        dxs[dxs < -0.5] = dxs[dxs < -0.5] + 1
        dys = ys[:,:,None,None] - ys[None,None,:,:]
        dys[dys > 0.5] = dys[dys > 0.5] - 1
        dys[dys < -0.5] = dys[dys < -0.5] + 1
        
    rng = np.random.default_rng(seed)
    
    eccs = rng.normal(size=shape)
    eccs = mean_ecc + std_ecc * eccs / np.std(eccs)
    eccs = np.clip(eccs,0.0,0.95)
    
    x_sizes = rng.normal(size=shape)
    x_sizes = (1 + x_sizes/np.std(x_sizes)*std_size)
    y_sizes = x_sizes*np.sqrt(1 - eccs**2)
    
    oris = rng.uniform(-np.pi/2,np.pi/2,size=shape)
    cos_oris = np.cos(oris)
    sin_oris = np.sin(oris)
    
    transf_dxs = (dxs*np.expand_dims(cos_oris,(-2,-1)) - dys*np.expand_dims(sin_oris,(-2,-1))) \
        * np.expand_dims(x_sizes,(-2,-1))
    transf_dys = (dxs*np.expand_dims(sin_oris,(-2,-1)) + dys*np.expand_dims(cos_oris,(-2,-1))) \
        * np.expand_dims(y_sizes,(-2,-1))
    
    return transf_dxs,transf_dys