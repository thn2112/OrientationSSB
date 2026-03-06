import numpy as np
from scipy.integrate import solve_ivp
from scipy import stats

class Model:
    def __init__(
        self,
        n_grid: int=20, # number of grid points per edge
        n_x: int=1, # number of LGN cells of each center type per grid point
        n_e: int=1, # expansion of excitatory cells per pixel
        n_i: int=1, # expansion of inhibitory cells per pixel
        s_x: float=0.08, # feedforward arbor decay length
        s_s: float=0.00, # retinotopic scatter decay length
        cut_lim: float=1.5, # arbor cutoff distance in terms of decay lengths
        flat_x: bool=True, # whether to use flat feedforward arbors
        flat_e: bool=False, # whether to use flat excitatory recurrent arbors
        flat_i: bool=False, # whether to use flat inhibitory recurrent arbors
        flat_s: bool=False, # whether to sample ret scat from uniform distribution
        gain_i: float=1.0, # gain of inhibitory cells
        hebb_wei: bool=False, # whether to use Hebbian learning for wei
        hebb_wii: bool=False, # whether to use Hebbian learning for wii
        prune: bool=False, # whether to prune weights
        rec_e_plast: bool=True, # whether recurrent weights are plastic
        rec_i_plast: bool=True, # whether recurrent weights are plastic
        rec_i_ltd: float=1.0, # factor for LTD of inhibitory weights
        w_prm_dict: dict=None, # dictionary of initial weight parameters
        init_dict: dict=None,
        seed: int=None,
        rx_wave_start: np.ndarray=None,
        ):
        # time constants of excitatory and inhibitory currents and NMDA vs AMPA fraction
        self.pow = 2
        
        # grid points and distances
        self.n_grid = n_grid
        self.n_egrid = n_e * n_grid
        self.txs,self.tys = np.meshgrid(np.linspace(0.5/self.n_grid,1-0.5/self.n_grid,self.n_grid),
                                        np.linspace(0.5/self.n_grid,1-0.5/self.n_grid,self.n_grid))
        self.txs,self.tys = self.txs.flatten(),self.tys.flatten()
        
        self.cxs,self.cys = np.meshgrid(np.linspace(0.5/self.n_egrid,1-0.5/self.n_egrid,self.n_egrid),
                                        np.linspace(0.5/self.n_egrid,1-0.5/self.n_egrid,self.n_egrid))
        self.cxs,self.cys = self.cxs.flatten(),self.cys.flatten()
        
        # retinotopic scatter
        if s_s < 1e-10:
            self.ret_scat_cxs = np.zeros_like(self.cxs)
            self.ret_scat_cys = np.zeros_like(self.cys)
        else:
            rng = np.random.default_rng(0)
            thtes = 2*np.pi*rng.uniform(0,1,size=self.cxs.size)
            if flat_s:
                rades = s_s*np.sqrt(rng.uniform(0,cut_lim,size=self.cxs.size))
            else:
                ray = stats.rayleigh()
                rades = s_s*ray.ppf(rng.uniform(size=self.cxs.size)*ray.cdf(cut_lim))
            self.ret_scat_cxs = rades*np.cos(thtes)
            self.ret_scat_cys = rades*np.sin(thtes)
        
        self.dists = np.sqrt(np.fmin(np.abs(self.cxs[:,None]-self.cxs[None,:]),
                                   1-np.abs(self.cxs[:,None]-self.cxs[None,:]))**2 +\
                             np.fmin(np.abs(self.cys[:,None]-self.cys[None,:]),
                                   1-np.abs(self.cys[:,None]-self.cys[None,:]))**2)
        self.scat_dists = np.sqrt(np.fmin(np.abs((self.cxs+self.ret_scat_cxs)[:,None]-self.txs[None,:]),
                                        1-np.abs((self.cxs+self.ret_scat_cxs)[:,None]-self.txs[None,:]))**2 +\
                                  np.fmin(np.abs((self.cys+self.ret_scat_cys)[:,None]-self.tys[None,:]),
                                        1-np.abs((self.cys+self.ret_scat_cys)[:,None]-self.tys[None,:]))**2)

        # number of cells
        self.n_e = self.n_egrid**2
        self.n_i = self.n_e
        self.n_4 = self.n_e + self.n_i
        self.n_lgn = 2*n_x*n_grid**2
        
        # define arbors
        broad_frac_e = w_prm_dict['broad_frac_e'] if w_prm_dict is not None else 1.0
        broad_frac_i = w_prm_dict['broad_frac_i'] if w_prm_dict is not None else 1.0
        s_n = w_prm_dict['s_n'] if w_prm_dict is not None else 0.08
        s_b = w_prm_dict['s_b'] if w_prm_dict is not None else 0.2
        if flat_x:
            self.ax = np.ones_like(self.scat_dists)
        else:
            self.ax = np.exp(-self.scat_dists**2/(2*s_x**2))
        self.ax = np.concatenate((self.ax,self.ax),axis=1)
        if flat_e:
            norm_n = np.fmax(1e-4,(self.dists <= cut_lim*s_n).sum(1).mean(0))
            norm_b = np.fmax(1e-4,(self.dists <= cut_lim*s_b).sum(1).mean(0))
            self.ae = ((self.dists <= cut_lim*s_n).astype(int) + broad_frac_e * norm_n/norm_b) *\
                np.ones_like(self.dists)
            self.mask_e = (self.dists <= cut_lim*s_b).astype(int)
        else:
            norm_n = np.fmax(1e-4,np.exp(-self.dists**2/(2*s_n**2)).sum(1).mean(0))
            norm_b = np.fmax(1e-4,np.exp(-self.dists**2/(2*s_b**2)).sum(1).mean(0))
            self.ae = np.exp(-self.dists**2/(2*s_n**2)) + broad_frac_e * norm_n/norm_b *\
                np.exp(-self.dists**2/(2*s_b**2))
            self.mask_e = (self.dists <= cut_lim*s_b).astype(int)
        if flat_i:
            norm_n = np.fmax(1e-4,(self.dists <= cut_lim*s_n).sum(1).mean(0))
            norm_b = np.fmax(1e-4,(self.dists <= cut_lim*s_b).sum(1).mean(0))
            self.ai = ((self.dists <= cut_lim*s_n).astype(int) + broad_frac_i * norm_n/norm_b) *\
                np.ones_like(self.dists)
            self.mask_i = (self.dists <= cut_lim*s_b).astype(int)
        else:
            norm_n = np.fmax(1e-4,np.exp(-self.dists**2/(2*s_n**2)).sum(1).mean(0))
            norm_b = np.fmax(1e-4,np.exp(-self.dists**2/(2*s_b**2)).sum(1).mean(0))
            self.ai = np.exp(-self.dists**2/(2*s_n**2)) + broad_frac_i * norm_n/norm_b *\
                np.exp(-self.dists**2/(2*s_b**2))
            self.mask_i = (self.dists <= cut_lim*s_b).astype(int)
        self.mask_x = (self.scat_dists <= cut_lim*s_x).astype(int)
        self.mask_x = np.concatenate((self.mask_x,self.mask_x),axis=1)
        np.place(self.ax,self.mask_x==0,0)
        np.place(self.ae,self.mask_e==0,0)
        np.place(self.ai,self.mask_i==0,0)
        self.ae /= np.mean(self.ae[self.mask_e==1])
        self.ai /= np.mean(self.ai[self.mask_i==1])
        
        self.n_x_in_arb = np.mean(np.sum(self.ax,axis=1),axis=0)
        self.n_e_in_arb = np.sum(self.ae,axis=1)[0]
        self.n_i_in_arb = np.sum(self.ai,axis=1)[0]
        print(self.n_x_in_arb,self.n_e_in_arb,self.n_i_in_arb)
        print(np.sum(self.mask_x,axis=1)[0],np.sum(self.mask_e,axis=1)[0],np.sum(self.mask_i,axis=1)[0])
        
        # postsynaptic weight normalization
        self.wff_sum = w_prm_dict['wff_sum'] if w_prm_dict is not None else 1.0
        self.wee_sum = w_prm_dict['wee_sum'] if w_prm_dict is not None else 0.05222626
        self.wie_sum = w_prm_dict['wie_sum'] if w_prm_dict is not None else 0.01108487
        self.wei_sum = w_prm_dict['wei_sum'] if w_prm_dict is not None else 0.07327422
        self.wii_sum = w_prm_dict['wii_sum'] if w_prm_dict is not None else 0.00822369
        # presynaptic weight normalization
        self.wlgn_sum = (self.n_e + self.n_i) * self.wff_sum / self.n_lgn
        self.w4e_sum = self.wee_sum + self.n_i/self.n_e * self.wie_sum
        self.w4i_sum = self.n_e/self.n_i * self.wei_sum + self.wii_sum
        
        print(self.wlgn_sum,self.w4e_sum,self.w4i_sum)

        # maximum allowed weights
        self.max_wff = 6*self.wff_sum / self.n_x_in_arb
        self.max_wee = 6*self.wee_sum / self.n_e_in_arb
        self.max_wei = 6*self.wei_sum / self.n_i_in_arb
        self.max_wie = 6*self.wie_sum / self.n_e_in_arb
        self.max_wii = 6*self.wii_sum / self.n_i_in_arb
        
        # whether to use Hebbian learning for wei and wii
        self.hebb_wei = hebb_wei
        self.hebb_wii = hebb_wii
        
        # whether to prune weights
        self.prune = prune
        
        # whether recurrent weights are plastic
        self.rec_e_plast = rec_e_plast
        self.rec_i_plast = rec_i_plast
        
        # factor for LTD of inhibitory weights
        self.rec_i_ltd = rec_i_ltd

        # RELU gains
        self.gain_e = 1.0
        self.gain_i = gain_i
        self.gain_mat = np.diag(np.concatenate((self.gain_e*np.ones(self.n_e),self.gain_i*np.ones(self.n_i))))
        
        self.dt_dyn = 0.01 # timescale for voltage dynamics
        self.a_avg = 1/60 # smoothing factor for average inputs
        self.targ_dw_rms = 0.002 # target root mean square weight change
        
        if init_dict is None:
            rng = np.random.default_rng(seed)
            
            # initialize weights
            self.wex = rng.uniform(0.2,0.8,size=(self.n_e,self.n_lgn)) * self.ax
            self.wix = rng.uniform(0.2,0.8,size=(self.n_i,self.n_lgn)) * self.ax
            self.wee = rng.uniform(0.2,0.8,size=(self.n_e,self.n_e)) * self.ae
            self.wei = rng.uniform(0.2,0.8,size=(self.n_e,self.n_i)) * self.ai
            self.wie = rng.uniform(0.2,0.8,size=(self.n_i,self.n_e)) * self.ae
            self.wii = rng.uniform(0.2,0.8,size=(self.n_i,self.n_i)) * self.ai
            
            # randomly choose some L4 cells to be more on/off dominated
            # on_dom = rng.choice([1,-1],size=(self.n_e,))
            # self.wex[:,:self.n_lgn//2] *= 1+0.3*on_dom[:,None]
            # self.wex[:,self.n_lgn//2:] *= 1-0.3*on_dom[:,None]
            # on_dom = rng.choice([1,-1],size=(self.n_i,))
            # self.wix[:,:self.n_lgn//2] *= 1+0.3*on_dom[:,None]
            # self.wix[:,self.n_lgn//2:] *= 1-0.3*on_dom[:,None]
            
            self.wex *= self.wff_sum / np.sum(self.wex,axis=1,keepdims=True)
            self.wix *= self.wff_sum / np.sum(self.wix,axis=1,keepdims=True)
            self.wee *= self.wee_sum / np.sum(self.wee,axis=1,keepdims=True)
            self.wei *= self.wei_sum / np.sum(self.wei,axis=1,keepdims=True)
            self.wie *= self.wie_sum / np.sum(self.wie,axis=1,keepdims=True)
            self.wii *= self.wii_sum / np.sum(self.wii,axis=1,keepdims=True)
            
            if self.prune: self.max_prop_thresh = 0.4
            else: self.max_prop_thresh = None
            if self.prune:
                self.prune_weights()
            
            # initialize learning rates
            self.wex_rate = 5e-6
            self.wix_rate = 5e-6
            self.wee_rate = 5e-6
            self.wei_rate = 5e-6
            self.wie_rate = 5e-6
            self.wii_rate = 5e-6
            
            # initialize average inputs and rates
            self.uee = np.zeros(self.n_e)
            self.uei = np.zeros(self.n_e)
            self.uie = np.zeros(self.n_i)
            self.uii = np.zeros(self.n_i)
            
            if rx_wave_start is None:
                rx_wave_start = np.ones(self.n_lgn)
            
            # calculate average inputs and rates at the start of a geniculate wave
            self.thresh = np.concatenate((np.ones(self.n_e)*np.mean(self.wex@rx_wave_start),
                                          np.ones(self.n_i)*np.mean(self.wix@rx_wave_start)))
            self.update_inps(rx_wave_start,100*self.dt_dyn,0.1)
            
            self.uee_avg = np.ones(self.n_e)*np.mean(self.uee)
            self.uei_avg = np.ones(self.n_e)*np.mean(self.uei)
            self.uie_avg = np.ones(self.n_i)*np.mean(self.uie)
            self.uii_avg = np.ones(self.n_i)*np.mean(self.uii)
            self.rx_avg = np.ones(self.n_lgn)*np.mean(rx_wave_start)
            
            x_sum = np.sum(self.wex,axis=0,keepdims=True) + np.sum(self.wix,axis=0,keepdims=True)
            e_sum = np.sum(self.wee,axis=0,keepdims=True) + np.sum(self.wie,axis=0,keepdims=True)
            i_sum = np.sum(self.wei,axis=0,keepdims=True) + np.sum(self.wii,axis=0,keepdims=True)
            print(np.mean(x_sum),np.mean(e_sum),np.mean(i_sum))
            
        else:
            self.wex = init_dict['wex']
            self.wix = init_dict['wix']
            self.wee = init_dict['wee']
            self.wei = init_dict['wei']
            self.wie = init_dict['wie']
            self.wii = init_dict['wii']
            self.wex_rate = init_dict['wex_rate']
            self.wix_rate = init_dict['wix_rate']
            self.wee_rate = init_dict['wee_rate']
            self.wei_rate = init_dict['wei_rate']
            self.wie_rate = init_dict['wie_rate']
            self.wii_rate = init_dict['wii_rate']
            self.uee = init_dict['uee']
            self.uei = init_dict['uei']
            self.uie = init_dict['uie']
            self.uii = init_dict['uii']
            self.uee_avg = init_dict['uee_avg']
            self.uei_avg = init_dict['uei_avg']
            self.uie_avg = init_dict['uie_avg']
            self.uii_avg = init_dict['uii_avg']
            self.rx_avg = init_dict['rx_avg']
            self.thresh = init_dict['thresh']
            self.max_prop_thresh = init_dict.get('max_prop_thresh',None)
            
        self.dwex = np.zeros_like(self.wex)
        self.dwix = np.zeros_like(self.wix)
        self.dwee = np.zeros_like(self.wee)
        self.dwei = np.zeros_like(self.wei)
        self.dwie = np.zeros_like(self.wie)
        self.dwii = np.zeros_like(self.wii)

    # voltage dynamics function
    def ode_func(
        self,
        t: float,
        u: np.ndarray,
        w: np.ndarray,
        h: np.ndarray,
        ):
        r = np.fmax(u,0)
        np.matmul(self.gain_mat,r,out=r)
        np.matmul(w,r,out=r)
        r[:] += h - u
        return r / self.dt_dyn

    # integrate rate dynamics and update inputs
    def update_inps(
        self,
        rx: np.ndarray,
        int_time: float,
        inh_mult: float=1,
        ):
        
        # calculate feedforward inputs
        he,hi = self.wex@rx,self.wix@rx
        h = np.concatenate((he,hi))
        
        # create full recurrent weight matrix
        w = np.block([[self.wee,-inh_mult*self.wei],[self.wie,-inh_mult*self.wii]])
        
        # integrate dynamics
        sol = solve_ivp(self.ode_func,[0,int_time],np.concatenate((self.uee-self.uei,self.uie-self.uii)),args=(w,h),t_eval=[int_time],method='RK23')
        
        # compute rates and update inputs
        r = np.fmax(sol.y[:,-1],0)
        np.matmul(self.gain_mat,r,out=r)
        re,ri = r[:self.n_e],r[self.n_e:]
        self.uee[:] = self.wee@re + he
        self.uie[:] = self.wie@re + hi
        self.uei[:] = self.wei@ri
        self.uii[:] = self.wii@ri
        
    # update input and rate averages
    def update_avgs(
        self,
        rx: np.ndarray,
        ):
        self.uee_avg[:] += self.a_avg * (self.uee - self.uee_avg)
        self.uei_avg[:] += self.a_avg * (self.uei - self.uei_avg)
        self.uie_avg[:] += self.a_avg * (self.uie - self.uie_avg)
        self.uii_avg[:] += self.a_avg * (self.uii - self.uii_avg)
        self.rx_avg[:] += self.a_avg * (rx - self.rx_avg)
        
        self.ue = self.uee - self.uei
        self.ui = self.uie - self.uii
        self.ue_avg = self.uee_avg - self.uei_avg
        self.ui_avg = self.uie_avg - self.uii_avg
        
    def reset_dw(
        self,
        ):
        self.dwex[:] = 0
        self.dwix[:] = 0
        self.dwee[:] = 0
        self.dwei[:] = 0
        self.dwie[:] = 0
        self.dwii[:] = 0
        
    # collect weight changes in a batch
    def collect_dw(
        self,
        rx: np.ndarray,
        ):
        
        self.dwex += self.ax * self.wex_rate * np.outer(self.ue - self.ue_avg,rx - self.rx_avg)
        self.dwix += self.ax * self.wix_rate * np.outer(self.ui - self.ui_avg,rx - self.rx_avg)
        if self.rec_e_plast:
            self.dwee += self.ae * self.wee_rate * np.outer(self.ue - self.ue_avg,self.ue - self.ue_avg)
            self.dwie += self.ae * self.wie_rate * np.outer(self.ui - self.ui_avg,self.ue - self.ue_avg)
        if self.rec_i_plast:
            if self.hebb_wei:
                self.dwei += self.ai * self.wei_rate * np.outer(self.ue - self.ue_avg,self.ui - self.ui_avg)
            else:
                self.dwei += self.ai * self.wei_rate * (np.outer(np.fmax(self.uei - self.uei_avg,0),
                                                                np.fmax(self.ui - self.ui_avg,0)) -\
                                                        self.rec_i_ltd*np.outer(np.fmax(self.ue - self.ue_avg,0),
                                                                np.fmax(self.ui - self.ui_avg,0)))
            if self.hebb_wii:
                self.dwii += self.ai * self.wii_rate * np.outer(self.ui - self.ui_avg,self.ui - self.ui_avg)
            else:
                self.dwii += self.ai * self.wii_rate * (np.outer(np.fmax(self.uii - self.uii_avg,0),
                                                                np.fmax(self.ui - self.ui_avg,0)) -\
                                                        self.rec_i_ltd*np.outer(np.fmax(self.ui - self.ui_avg,0),
                                                                np.fmax(self.ui - self.ui_avg,0)))

    def sum_norm_dw(
        self,
        ):
        norm = np.sum(self.ax,axis=0,keepdims=True) + np.sum(self.ax,axis=0,keepdims=True)
        norm = np.where(norm==0,1,norm)
        eps = (np.sum(self.dwex,axis=0,keepdims=True) + np.sum(self.dwix,axis=0,keepdims=True)) / norm
        self.dwex -= eps * self.ax
        self.dwix -= eps * self.ax
        # self.dwee -= np.mean(self.dwee,axis=1,keepdims=True)
        # self.dwei -= np.mean(self.dwei,axis=1,keepdims=True)
        # self.dwie -= np.mean(self.dwie,axis=1,keepdims=True)
        # self.dwii -= np.mean(self.dwii,axis=1,keepdims=True)
        
    # update learning rates
    def update_learn_rates(
        self,
        ):
        self.wex_rate *= self.targ_dw_rms / np.sqrt(np.mean(np.extract(self.mask_x==1,self.dwex**2)))
        self.wix_rate *= self.targ_dw_rms / np.sqrt(np.mean(np.extract(self.mask_x==1,self.dwix**2)))
        if self.rec_e_plast:
            self.wee_rate *= self.targ_dw_rms / np.sqrt(np.mean(np.extract(self.mask_e==1,self.dwee**2)))
            self.wie_rate *= self.targ_dw_rms / np.sqrt(np.mean(np.extract(self.mask_e==1,self.dwie**2)))
        if self.rec_i_plast:
            self.wei_rate *= self.targ_dw_rms / np.sqrt(np.mean(np.extract(self.mask_i==1,self.dwei**2)))
            self.wii_rate *= self.targ_dw_rms / np.sqrt(np.mean(np.extract(self.mask_i==1,self.dwii**2)))
        # self.wex_rate = self.targ_dw_rms / np.sqrt(np.mean(self.dwex**2))
        # self.wix_rate = self.targ_dw_rms / np.sqrt(np.mean(self.dwix**2))
        # self.wee_rate = self.targ_dw_rms / np.sqrt(np.mean(self.dwee**2))
        # self.wei_rate = self.targ_dw_rms / np.sqrt(np.mean(self.dwei**2))
        # self.wie_rate = self.targ_dw_rms / np.sqrt(np.mean(self.dwie**2))
        # self.wii_rate = self.targ_dw_rms / np.sqrt(np.mean(self.dwii**2))
        
        # self.dwex *= self.wex_rate
        # self.dwix *= self.wix_rate
        # self.dwee *= self.wee_rate
        # self.dwei *= self.wei_rate
        # self.dwie *= self.wie_rate
        # self.dwii *= self.wii_rate
        if np.isnan(self.wex_rate):
            self.wex_rate = 1e-6
        if np.isnan(self.wix_rate):
            self.wix_rate = 1e-6
        if self.rec_e_plast:
            if np.isnan(self.wee_rate) or np.isinf(self.wee_rate):
                self.wee_rate = 1e-6
            if np.isnan(self.wie_rate) or np.isinf(self.wie_rate):
                self.wie_rate = 1e-6
        if self.rec_i_plast:
            if np.isnan(self.wei_rate) or np.isinf(self.wei_rate):
                self.wei_rate = 1e-6
            if np.isnan(self.wii_rate) or np.isinf(self.wii_rate):
                self.wii_rate = 1e-6
        
    def prune_weights(
        self,
        ):
        # update pruning threshold
        max_norm_wex = self.wex/np.max(self.wex,axis=1,keepdims=True)
        max_norm_wex = max_norm_wex[max_norm_wex > 1e-5 / np.mean(np.max(self.wex,axis=1))]
        max_norm_wix = self.wix/np.max(self.wix,axis=1,keepdims=True)
        max_norm_wix = max_norm_wix[max_norm_wix > 1e-5 / np.mean(np.max(self.wix,axis=1))]
        new_thresh = np.quantile(np.concatenate((max_norm_wex,max_norm_wix)),0.6)
        self.max_prop_thresh += self.a_avg * (new_thresh - self.max_prop_thresh)
        # print("new threshold:",self.max_prop_thresh)
        
        # implement pruning by shrinking small weights
        thresh = np.max(self.wex,axis=1,keepdims=True) * self.max_prop_thresh
        self.wex *= np.heaviside(self.wex,0)*(0.9+0.1*np.heaviside(self.wex-thresh,0))
        self.wex *= self.wff_sum / np.sum(self.wex,axis=1,keepdims=True)
        
        thresh = np.max(self.wix,axis=1,keepdims=True) * self.max_prop_thresh
        self.wix *= np.heaviside(self.wix,0)*(0.9+0.1*np.heaviside(self.wix-thresh,0))
        self.wix *= self.wff_sum / np.sum(self.wix,axis=1,keepdims=True)
        
    # update weights with collected changes, then clip and normalize weights
    def update_weights(
        self,
        ):
        self.wex += self.dwex
        self.wix += self.dwix
        if self.rec_e_plast:
            self.wee += self.dwee
            self.wie += self.dwie
        if self.rec_i_plast:
            self.wei += self.dwei
            self.wii += self.dwii
        
        if self.prune:
            self.prune_weights()
        
        # alternate clipping, presynaptic normalization, and postsynaptic normalization
        for _ in range(4):
            # clip weights
            np.clip(self.wex,1e-10,self.max_wff*self.ax,out=self.wex)
            np.clip(self.wix,1e-10,self.max_wff*self.ax,out=self.wix)
            if self.rec_e_plast:
                np.clip(self.wee,1e-10,self.max_wee*self.ae,out=self.wee)
                np.clip(self.wie,1e-10,self.max_wie*self.ae,out=self.wie)
            if self.rec_i_plast:
                np.clip(self.wei,1e-10,self.max_wei*self.ai,out=self.wei)
                np.clip(self.wii,1e-10,self.max_wii*self.ai,out=self.wii)
            
            # presynaptic normalization
            x_sum = np.sum(self.wex,axis=0,keepdims=True) + np.sum(self.wix,axis=0,keepdims=True)
            self.wex *= self.wlgn_sum / x_sum
            self.wix *= self.wlgn_sum / x_sum
            if self.rec_e_plast:
                e_sum = np.sum(self.wee,axis=0,keepdims=True) + np.sum(self.wie,axis=0,keepdims=True)
                self.wee *= self.w4e_sum / e_sum
                self.wie *= self.w4e_sum / e_sum
            if self.rec_i_plast:
                i_sum = np.sum(self.wei,axis=0,keepdims=True) + np.sum(self.wii,axis=0,keepdims=True)
                self.wei *= self.w4i_sum / i_sum
                self.wii *= self.w4i_sum / i_sum
            
            # clip weights
            np.clip(self.wex,1e-10,self.max_wff*self.ax,out=self.wex)
            np.clip(self.wix,1e-10,self.max_wff*self.ax,out=self.wix)
            if self.rec_e_plast:
                np.clip(self.wee,1e-10,self.max_wee*self.ae,out=self.wee)
                np.clip(self.wie,1e-10,self.max_wie*self.ae,out=self.wie)
            if self.rec_i_plast:
                np.clip(self.wei,1e-10,self.max_wei*self.ai,out=self.wei)
                np.clip(self.wii,1e-10,self.max_wii*self.ai,out=self.wii)
            
            # postsynaptic normalization
            self.wex *= self.wff_sum / np.sum(self.wex,axis=1,keepdims=True)
            self.wix *= self.wff_sum / np.sum(self.wix,axis=1,keepdims=True)
            if self.rec_e_plast:
                self.wee *= self.wee_sum / np.sum(self.wee,axis=1,keepdims=True)
                self.wie *= self.wie_sum / np.sum(self.wie,axis=1,keepdims=True)
            if self.rec_i_plast:
                self.wei *= self.wei_sum / np.sum(self.wei,axis=1,keepdims=True)
                self.wii *= self.wii_sum / np.sum(self.wii,axis=1,keepdims=True)
