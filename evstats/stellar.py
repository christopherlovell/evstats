import numpy as np
from scipy.stats import uniform,norm,expon,truncnorm,lognorm,gaussian_kde


def apply_fs_distribution(pdf, _x, method='lognormal', _N=int(1e4), f_b=0.16):
    """
    Use Monte Carlo sampling to estimate the combined pdf of an EVS distribution and an f_s distribution.
    
    First, sample from EVS PDF (phi), then sample from f_s PDF (uniform, gaussian...). 
    Multiply together to get distribution of product phi * f_s .
    See: https://stackoverflow.com/questions/29095070/how-to-simulate-from-an-arbitrary-continuous-probability-distribution
    
    Parameters
    ----------
    pdf (array): probability density function on x
    x (array): coordinates of pdf
    method (str): parametric form of f_s pdf, one of 'lognormal', 'normal', 'uniform' or 'exponential'
    _N (int): number of MC samples
    f_b (float): additional normalisation constant to apply (e.g. baryon fraction)
    
    Returns
    -------
    pdf (array): product of evs and f_s pdfs
    
    """
    
    ## stellar mass CIs
    cumpdf = np.cumsum(pdf) / np.cumsum(pdf)[-1]
    randv = np.random.uniform(size=_N)
    idx1 = np.searchsorted(cumpdf, randv)
    idx0 = np.where(idx1==0, 0, idx1-1)
    idx1[idx0==0] = 1  # force first index if at edge of domain
    frac1 = (randv - cumpdf[idx0]) / (cumpdf[idx1] - cumpdf[idx0])
    randdist = _x[idx0]*(1-frac1) + _x[idx1]*frac1  # random samples from halo EVS PDF
    
    ## sample from f_s distribution
    if method=='normal': 
        myclip_a = 0; myclip_b = 1; my_mean = 0.2; my_std = 0.1 ## params for truncated log norm
        _a, _b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
        f_s = truncnorm.rvs(_a, _b, loc = my_mean, scale = my_std, size=_N)
    elif method=='uniform': f_s = uniform.rvs(size=_N, loc=0, scale=1)  # U[0,1]
    elif method=='exponential': f_s = expon.rvs(size=_N, scale=0.1) 
    elif method=='lognormal': f_s = _trunc_lognormal(-2, 1, N=_N) 
    else: raise ValueError("No valid method provided");
    
    mstar_samp = np.log10(10**randdist * f_s * f_b) # samples from mstar
    kernel = gaussian_kde(mstar_samp, bw_method=0.08)
    return kernel.pdf(_x)


def _trunc_lognormal(mean,sigma,N,lolim=0,hilim=1):
    x = np.zeros(N)
    _outside = np.ones(N, dtype=bool)
    while np.sum(_outside) > 0:
        x[_outside] = lognorm.rvs(s=sigma, scale=np.exp(mean), size=np.sum(_outside))
        # x[_outside] = np.random.lognormal(mean=mean, sigma=sigma, size=np.sum(_outside))
        _outside = (x < lolim) | (x >= hilim)
            
    return x
