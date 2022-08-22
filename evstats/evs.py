import sys
import numpy as np

from scipy import integrate

import hmf
import astropy

import matplotlib.pyplot as plt

def evs_hypersurface_pdf(mf = hmf.MassFunction(), V = 33510.321):
    """
    Calculate extreme value probability density function for the dark matter
    halo population on a spatial hypersurface (fixed redshift).

    Parameters
    ----------
    mf : mass function, from `hmf` package
    V : volume (default: a sphere with radius 20 Mpc)

    Returns
    -------
    phi:

    """

    n_tot = integrate.trapz(mf.dndlog10m, np.log10(mf.m))
    f = mf.dndlog10m[:-1] / n_tot
    F = integrate.cumtrapz(mf.dndlog10m, np.log10(mf.m)) / n_tot
    N = V*n_tot
    phi_max = N*f*(F**(N-1))
    return phi_max



def evs_bin_pdf(mf = hmf.MassFunction(), zmin=0., zmax=0.1, dz=0.01, mmin=12, mmax=18, dm = 0.01, fsky=1.):
    """
    Calculate EVS in redshift and mass bin

    Parameters
    ----------
    zmin : z minimum
    zmax : z maximum
    dz: delta z
    mmin: mass minimum (log10 (h^{-1} M_{\sol}) )
    mmax: mass maximum (log10 (h^{-1} M_{\sol}) )
    dm: delta m (log10 (h^{-1} M_{\sol}) )
    fsky: fraction of sky

    Returns
    -------
    phi: probability density function
    ln10m_range: corresponding mass values for PDF (log10 (h^{-1} M_{\sol}) )
    
    """

    N, f, F, ln10m_range = _evs_bin(mf=mf, zmin=zmin, zmax=zmax, dz=dz, mmin=mmin, mmax=mmax, dm=dm)

    phi =_apply_fsky(N, f, F, fsky)

    return phi, ln10m_range


def _apply_fsky(N, f, F, fsky):     
    N_sky = N * fsky
    f_sky = f / N_sky
    f_sky *= fsky
    
    F_sky = F / N_sky
    F_sky *= fsky

    return N_sky * f_sky * pow(F_sky, N_sky - 1.)
    
    
def _evs_bin(mf = hmf.MassFunction(), zmin=0., zmax=0.1, dz=0.01, mmin=12, mmax=18, dm = 0.01):
    """
    Calculate EVS (ignoring fsky dependence). Worker function for `evs_bin_pdf`

    Parameters
    ----------
    zmin : z minimum
    zmax : z maximum
    dz: delta z
    mmin: mass minimum (log10 (h^{-1} M_{\sol}) )
    mmax: mass maximum (log10 (h^{-1} M_{\sol}) )
    dm: delta m (log10 (h^{-1} M_{\sol}) )

    Returns
    -------
    N (float)
    f (array)
    F (array)
    ln10m_range (array)
    
    """

    mf.update(Mmin=mmin, Mmax=mmax, dlog10m=dm)

    N = _computeNinbin(mf=mf, zmin=zmin, zmax=zmax, dz=dz)

    # need to set lower limit slightly higher otherwise hmf complains.
    # should have no impact on F if set sufficiently low.
    ln10m_range = np.log10(mf.m[np.log10(mf.m) >= mmin+1])

    F = np.array([_computeNinbin(mf=mf, zmin=zmin, zmax=zmax, lnmax=lnmax, dz=dz) \
     for lnmax in ln10m_range])

    f = np.gradient(F, mf.dlog10m)
    
    return N, f, F, ln10m_range


def _computeNinbin(mf, zmin, zmax, lnmax=False, dz=0.01):

    if lnmax: mf.update(Mmax=lnmax)

    zees = np.arange(zmin, zmax, dz, dtype='longdouble')  # z range

    # calculate dvdz in advance to take advantage of vectorization
    dvdz = mf.cosmo.differential_comoving_volume(zees).value.astype('longdouble') * (4. * np.pi)

    dndmdz = [_dNdlnmdz(z=z, mf=mf, dvdz=dv) for z, dv in zip(zees,dvdz)]
    
    # integrate over z
    return integrate.trapz(dndmdz, zees)


def _dNdlnmdz(z, mf, dvdz):
    mf.update(z=z)
    return integrate.trapz(mf.dndlnm.astype('longdouble') * dvdz, np.log(mf.m))


if __name__ == '__main__':

    # initialise mass function
    mass_function = hmf.MassFunction()

    # set cosmology using astropy
    from astropy.cosmology import Planck13
    mass_function.cosmo_model = Planck13

    # set redshift
    mass_function.z = 0.0

    # set mass range and resolution
    mass_function.dlog10m = 0.1
    mass_function.Mmin = 12
    mass_function.Mmax = 18

    phi_max = evs_hypersurface_pdf(mf = mass_function)

    plt.loglog(mass_function.m[:-1], phi_max)
    plt.ylim(10**-4,1)
    plt.xlabel('Mass $[M_{\odot}h^{-1}]$')
    plt.ylabel('$\phi(M_{max})$')

    plt.show()
