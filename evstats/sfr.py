import numpy as np
import matplotlib.pyplot as plt

from scipy import integrate
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


def _sfr_at_age(sfh, age):
    """
    Evaluate a star formation history at a set of stellar ages.

    Parameters
    ----------
    sfh : object
        One of: a parametric SFH with a `get_sfr(age)` method (e.g. any of
        the `synthesizer.parametric.SFH` forms), a callable of age in years,
        or a binned history given as a tuple `(bins, sfr)`. Bins are either
        edges (`len(bins) == len(sfr) + 1`, piecewise constant) or the ages
        at which the SFR is tabulated (linearly interpolated).
    age : array-like
        Stellar ages in years, measured back from the epoch of observation.

    Returns
    -------
    sfr : ndarray
        SFR at each age, zero outside the range covered by the SFH.
    """
    if hasattr(sfh, 'get_sfr'):
        return np.asarray(sfh.get_sfr(np.asarray(age, dtype=float)), dtype=float)

    if callable(sfh):
        return np.asarray(sfh(np.asarray(age, dtype=float)), dtype=float)

    bins, sfr = (np.asarray(_a, dtype=float) for _a in sfh)

    if len(bins) == len(sfr) + 1:
        idx = np.searchsorted(bins, age, side='right') - 1
        inside = (idx >= 0) & (idx < len(sfr))
        return np.where(inside, sfr[np.clip(idx, 0, len(sfr) - 1)], 0.)

    return np.interp(age, bins, sfr, left=0., right=0.)


def mass_growth_track(sfh, z_obs, log10_mstar, z=None, cosmo=Planck15, n_t=1000):
    """
    Project an observed stellar mass back in redshift using an assumed SFH.

    The SFH is integrated from the beginning of star formation up to the
    epoch of observation, and the cumulative mass formed is normalised so
    that it matches the observed stellar mass at `z_obs`. The track is
    therefore independent of the SFH normalisation, and only its shape (and
    duration) matters. Mass loss from stellar evolution is not modelled, so
    the track is the mass *formed* rescaled to the observed mass.

    Parameters
    ----------
    sfh : object
        Star formation history, in any of the forms accepted by
        `_sfr_at_age` (parametric, callable or binned).
    z_obs : float
        Redshift at which the galaxy is observed.
    log10_mstar : float
        Observed stellar mass, log10(M / Msun).
    z : array-like, optional
        Redshifts at which to evaluate the track. Defaults to 200 points
        between `z_obs` and z = 20.
    cosmo : astropy.cosmology instance, optional
        Cosmology used to convert between redshift and cosmic time.
    n_t : int, optional
        Number of samples used to integrate the SFH.

    Returns
    -------
    z : ndarray
        Redshifts of the track.
    log10_mstar_z : ndarray
        log10(M / Msun) at each redshift, -inf before star formation begins.
    """
    if z is None:
        z = np.linspace(z_obs, 20., 200)
    z = np.asarray(z, dtype=float)

    t_obs = cosmo.age(z_obs).to_value('yr')
    t = np.linspace(0., t_obs, n_t)

    # SFH ages run backwards from the epoch of observation
    sfr = _sfr_at_age(sfh, t_obs - t)
    m = integrate.cumulative_trapezoid(sfr, t, initial=0.)

    if m[-1] <= 0:
        raise ValueError('SFH forms no mass before the epoch of observation')

    m /= m[-1]
    m_z = np.interp(cosmo.age(z).to_value('yr'), t, m, left=0.)

    with np.errstate(divide='ignore'):
        return z, np.log10(10**log10_mstar * m_z)


def plot_mass_growth_track(
    sfh,
    z_obs,
    log10_mstar,
    log10_mstar_err=None,
    z=None,
    ax=None,
    cosmo=Planck15,
    color='black',
    alpha=0.3,
    **kwargs
):
    """
    Plot the redshift evolution of the stellar mass implied by an SFH.

    Parameters
    ----------
    sfh, z_obs, log10_mstar, z, cosmo :
        As in `mass_growth_track`.
    log10_mstar_err : float or (2,) array-like, optional
        Uncertainty on the observed mass in dex, either symmetric or as
        (lower, upper). Shaded as a constant offset about the track.
    ax : matplotlib axis, optional
        Axis to plot on. A new figure is created if not provided.
    color, alpha : optional
        Colour of the track, and opacity of the shaded uncertainty.
    **kwargs :
        Passed to `ax.plot`.

    Returns
    -------
    ax : matplotlib axis
    """
    if ax is None:
        _, ax = plt.subplots()

    z, log10_mstar_z = mass_growth_track(
        sfh, z_obs, log10_mstar, z=z, cosmo=cosmo,
    )

    ax.plot(z, log10_mstar_z, color=color, **kwargs)

    if log10_mstar_err is not None:
        lo, hi = np.broadcast_to(log10_mstar_err, 2)
        ax.fill_between(z, log10_mstar_z - lo, log10_mstar_z + hi,
                        color=color, alpha=alpha, lw=0)

    return ax
