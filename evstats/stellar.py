import numpy as np
from scipy.stats import uniform,norm,expon,lognorm,gaussian_kde


def sample_halo_evs_pdf(pdf, _x, _N):
    cumpdf = np.cumsum(pdf) / np.cumsum(pdf)[-1]
    randv = np.random.uniform(size=_N)
    idx1 = np.searchsorted(cumpdf, randv)
    idx0 = np.where(idx1==0, 0, idx1-1)
    idx1[idx0==0] = 1  # force first index if at edge of domain
    frac1 = (randv - cumpdf[idx0]) / (cumpdf[idx1] - cumpdf[idx0])
    randdist = _x[idx0]*(1-frac1) + _x[idx1]*frac1  # random samples from halo EVS PDF
    return randdist


def log10_fs_pdf(v, method='lognormal'):
    """
    PDF of v = log10(f_s), for the f_s distributions truncated to 0 < f_s <= 1.

    Parameters
    ----------
    v (array): log10 stellar fraction, uniformly spaced and <= 0
    method (str): parametric form of f_s pdf, one of 'lognormal', 'normal', 'uniform' or 'exponential'

    Returns
    -------
    (array): normalised pdf of v

    """
    f_s = 10**v

    if method=='normal': p = norm.pdf(f_s, loc=0.2, scale=0.1)
    elif method=='uniform': p = uniform.pdf(f_s)
    elif method=='exponential': p = expon.pdf(f_s, scale=0.1)
    elif method=='lognormal': p = lognorm.pdf(f_s, s=1, scale=np.exp(-2))
    else: raise ValueError("No valid method provided");

    p = p * f_s * np.log(10)  # Jacobian, df_s / dv
    return p / np.trapezoid(p, v)  # truncation to f_s <= 1 by renormalisation


def apply_fs_distribution(pdf, _x, method='lognormal', f_b=0.16, vmin=-8.):
    """
    Combine an EVS distribution with an f_s distribution to give the stellar mass PDF.

    The stellar mass is the product M_star = f_b * f_s * M_halo, which in log
    space is a sum, so the PDF is the convolution of the halo EVS PDF with the
    PDF of log10(f_s), shifted by the baryon fraction.

    Parameters
    ----------
    pdf (array): probability density function on x
    _x (array): log10 halo mass coordinates of pdf, uniformly spaced
    method (str): parametric form of f_s pdf, see `log10_fs_pdf`
    f_b (float): additional normalisation constant to apply (e.g. baryon fraction)
    vmin (float): lower limit on log10(f_s) to include in the convolution

    Returns
    -------
    pdf (array): stellar mass pdf, on the same coordinates _x

    """
    dx = _x[1] - _x[0]
    v = np.arange(vmin, dx / 2, dx)  # log10(f_s), up to and including zero
    kernel = log10_fs_pdf(v, method)

    conv = np.convolve(pdf, kernel) * dx  # starts at _x[0] + v[0] + log10(f_b)
    i0 = int(round((-v[0] - np.log10(f_b)) / dx))
    conv = np.append(conv, np.zeros(max(0, i0 + len(_x) - len(conv))))
    return conv[i0:i0 + len(_x)]


def _trunc_lognormal(mean,sigma,N,lolim=0,hilim=1):
    x = np.zeros(N)
    _outside = np.ones(N, dtype=bool)
    while np.sum(_outside) > 0:
        x[_outside] = lognorm.rvs(s=sigma, scale=np.exp(mean), size=np.sum(_outside))
        # x[_outside] = np.random.lognormal(mean=mean, sigma=sigma, size=np.sum(_outside))
        _outside = (x < lolim) | (x >= hilim)
            
    return x


def halo_dependent_fs(
    halom,
    lo_halo_mass=13.5,
    hi_halo_mass=14.0,
    f_b=0.16,
):
    """
    Apply Andreon+10 fits:
    https://ui.adsabs.harvard.edu/abs/2010MNRAS.407..263A/abstract
    """
    _N = len(halom)
 
    # Sample parameters of the stellar-halo mass relation 
    slope = norm.rvs(loc=0.45, scale=0.08, size=_N)
    intersect = norm.rvs(loc=12.68, scale=0.03, size=_N)

    # Convert halo masses to stellar fractions
    f_s_hi = 10**((halom - 14.5) * slope + intersect) / 10**halom

    # Apply truncated lognorm at low halo masses
    f_s_lo = _trunc_lognormal(-2, 1, N=_N)

    f_s = f_s_hi.copy()

    # For haloes in the transition region, use mixture
    mask = (halom > lo_halo_mass) & (halom < hi_halo_mass)
    weighting = (halom[mask] - lo_halo_mass) / (hi_halo_mass - lo_halo_mass)
    f_s[mask] = (weighting * f_s_hi[mask] + (1 - weighting) * f_s_lo[mask])

    mask = (halom < lo_halo_mass)
    f_s[mask] = f_s_lo[mask]

    # Apply stellar and baryon fraction
    log_mstar = np.log10(10**halom * f_s * f_b)

    return log_mstar, f_s


def apply_halo_dependent_fs(
    pdf,
    log10m,
    lo_halo_mass=13.5,
    hi_halo_mass=14.0,
    N=int(1e3),
    f_b=0.16
):
    # Sample _N haloes
    halom = sample_halo_evs_pdf(pdf, log10m, N)
     
    log_mstar, _ = halo_dependent_fs(halom, f_b=f_b)
   
    kernel = gaussian_kde(log_mstar.flatten(), bw_method=0.08)
    return kernel.pdf(log10m)
 
