import numpy as np
from scipy import integrate
import hmf

def compute_conf_ints(pdf, x, lims = [0.0013498980, 0.0227501319, 0.15865525, 
                                      0.500, 0.8413447, 0.97724986, 0.998650102]):
    """
    Compute confidence intervals for a given pdf
    
    Parameters
    ----------
    x (array): x coordinates
    pdf (array): probability density function evaluated at x
    lims (array): confidence intervals to compute

    Returns
    -------
    CI (array): confidence intervals
    
    """
    
    intcum = integrate.cumtrapz(pdf, x, initial=0.)
    
    if np.squeeze(pdf).ndim > 1:
        CI = np.vstack([x[[np.max(np.where(_ic < lim)) \
                         for _ic in intcum]] for lim in lims]).T
    else:
        CI = x[[np.max(np.where(intcum < lim)) for lim in lims]]
        
    return CI




def eddington_bias(m, m_err, mf = hmf.MassFunction()):
    """
    Calculate masses corrected for eddington bias.
    
    ln M_corrected = ln M_observed + 0.5 * epsilon * sigma_ln_M**2
    
    where epsilon is the slope of the halo mass function.
    
    Args:
    m (array): object masses in log base 10
    m_err (array): object mass errors in log base 10
    
    Returns:
    (array): corrected masses 
    """
    m_err_ln = np.log(10**(m+np.mean(m_err,axis=0))) - np.log(10**m)
    
    epsilon = np.zeros(len(m))
    for i,_m in enumerate(m):
        mf.Mmin = _m #-1e-1
        epsilon[i] = ((np.diff(np.log(mf.dndlnm))) / np.diff(np.log(mf.m)))[0]
    
    return np.log10(np.exp(np.log(10**m) + (0.5 * epsilon * m_err_ln**2))), epsilon
