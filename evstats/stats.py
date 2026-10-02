import numpy as np
from scipy import integrate
import hmf

def compute_conf_ints(pdf, x, y=None, lims = [0.0013498980, 0.0227501319, 0.15865525,
                                              0.500, 0.8413447, 0.97724986, 0.998650102]):
    """
    Compute confidence intervals for a given pdf

    Parameters
    ----------
    x (array): x coordinates
    pdf (array): probability density function evaluated at x
    y (array): optional monotonic function of x, evaluated at x. If given the
        intervals are returned in these coordinates instead, e.g. the flux at
        each stellar mass. The cdf is always integrated over x, so no Jacobian
        is needed; quantiles are preserved under a monotonic transform.
    lims (array): confidence intervals to compute

    Returns
    -------
    CI (array): confidence intervals

    """

    pdf = np.asarray(pdf, dtype=float)
    cdf = integrate.cumulative_trapezoid(pdf, x, axis=-1, initial=0.)
    coord = x if y is None else np.asarray(y, dtype=float)

    # Invert each (per-row normalised) CDF by interpolation, so the CI
    # coordinates are continuous rather than snapped to the grid.
    if np.squeeze(pdf).ndim > 1:
        norm = cdf[:, -1][:, None]
        cdf = np.divide(cdf, norm, out=np.zeros_like(cdf), where=norm > 0)
        rows = coord if coord.ndim > 1 else np.broadcast_to(coord, cdf.shape)
        return np.array([np.interp(lims, _c, _y) for _c, _y in zip(cdf, rows)])

    if cdf[-1] == 0:
        return np.zeros(len(lims))
    return np.interp(lims, cdf / cdf[-1], coord)


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
