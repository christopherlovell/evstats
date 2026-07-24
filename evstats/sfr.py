import numpy as np

from unyt import Msun
from astropy.cosmology import Planck15

from synthesizer.parametric import Stars


def halo_mass_accretion_rate(mvir, z, cosmo=Planck15):
    """
    Median halo mass accretion rate dMvir/dt from Yung+24 (Sect. 3.3; Eq. 5).

    Parameters
    ----------
    mvir : float or array-like
        Halo virial mass in solar masses (physical units, not h^{-1}).
    z : float or array-like
        Redshift.
    cosmo : astropy.cosmology instance, optional
        Cosmology used to compute E(z) = H(z) / H0. Defaults to Planck15.

    Returns
    -------
    rate : ndarray
        Mass accretion rate dMvir/dt in solar masses per year.

    Notes
    -----
    Coefficients assume log10 for beta(z); E(z) is the dimensionless
    H(z)/H0 from the supplied astropy cosmology.
    """
    z_arr = np.asarray(z)
    a = 1.0 / (1.0 + z_arr)
    alpha = 0.858 + 1.554 * a - 1.176 * a**2
    log10_beta = 2.578 - 0.989 * a - 1.545 * a**2
    beta = 10.0 ** log10_beta

    e_z = cosmo.efunc(z_arr)
    m12 = np.asarray(mvir) / 1e12

    return beta * np.power(m12 * e_z, alpha)


def average_sfr_over_window(sfh, total_mass, window_myr, metal_dist):
    """
    Average SFR from t=0 to `window_myr` using a Synthesizer Stars object built 
    from the SFH.

    Parameters
    ----------
    sfh : synthesizer.parametric.SFH instance
        Star formation history object (e.g. SFH.Exponential).
    total_mass : float
        Total stellar mass formed up to the current age (Msun). SFH
        normalisations are typically per unit initial mass, so this
        rescales to Msun/yr.
    window_myr : float
        Width of the time window, starting at t=0, over which to average
        the SFR (Myr).
    metal_dist : synthesizer.parametric.ZDist instance
        Metallicity distribution function for the Stars object.

    Returns
    -------
    sfr_avg : float
        Mean SFR over [0, window_myr] in Msun/yr (per unit mass if
        sfh.sfr is per unit mass).
    """
    if window_myr <= 0:
        return np.nan    

    log10ages = np.linspace(6.0, 10.3, 300)
    metallicities = np.linspace(1e-4, 0.03, 20)

    stars = Stars(
        np.asarray(log10ages, dtype=float),
        np.asarray(metallicities, dtype=float),
        sf_hist=sfh,
        metal_dist=metal_dist,
        initial_mass=total_mass * Msun,
    )

    sf_hist = stars.get_sfh()  # Msol
    ages_myr = (10.0 ** np.asarray(stars.log10ages, dtype=float)) / 1e6
    mask = ages_myr <= window_myr
    if not np.any(mask):
        return np.nan

    mass_window = np.sum(sf_hist[mask])
    sfr_avg = mass_window / (window_myr * 1e6)
    return sfr_avg
