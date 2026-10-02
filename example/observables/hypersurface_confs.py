import numpy as np
import h5py
import matplotlib.pyplot as plt

import astropy.units as u
from evstats import evs
from evstats.stats import compute_conf_ints
from evstats.stellar import apply_fs_distribution


with h5py.File('../data/evs_all.h5', 'r') as hf:
    log10m = hf['log10m'][:]
    f = hf['f'][:]
    F = hf['F'][:]
    N = hf['N'][:]
    z = hf['z'][:]

_obs_str = 'confs'

whole_sky = (41252.96 * u.deg**2).to(u.arcmin**2)
survey_area = 0.28 * u.deg**2
fsky = float(survey_area / whole_sky)
phi_max = evs._apply_fsky(N, f, F, fsky)
redshift_idx = np.arange(len(z))

# Stellar masses from the baryon fraction and a lognormal stellar fraction.
f_b = 0.16
mstar_pdf = np.vstack([apply_fs_distribution(p, log10m, f_b=f_b) for p in phi_max])
# Conservative limit: all baryons converted into stars (f_star = 1), +3 sigma.
mstar_fs1 = compute_conf_ints(phi_max, log10m)[:, 6] + np.log10(f_b)

# Fiducial model: exponential SFH with BPASS + nebular emission
sfh_tag = "Exponential_tau-0.1"
sps_tag = f"{sfh_tag}_bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-c23.01-sps"

bands = ['NIRCam.F150W', 'NIRCam.F277W', 'NIRCam.F444W']
_greens = plt.get_cmap('Greens')
colors = [_greens(0.7), _greens(0.45), _greens(0.25)]

# Casey+24 COSMOS-Web candidates; redshifts from BAGPIPES, aperture-based photometry
z_obs = np.array([9.69, 12.08, 11.46, 11.46, 9.15, 9.82, 10.20, 11.19, 10.63, 13.2, 13.4, 14.0])
zerr = np.array([[0.25, 0.16, 0.04, 0.23, 0.35, 0.45, 0.54, 0.3, 0.46, 0.9, 1.2, 2.4],
                 [0.24, 0.13, 0.43, 0.28, 0.29, 0.22, 0.51, 0.31, 0.52, 0.6, 0.7, 1.1]])

obs_data = {
    'NIRCam.F150W': (
        np.array([60.1, 12.1, 26.2, 18.7, 22.5, 37.1, 24.3, 20.6, 20.8, 0.0, 2.4, 3.4]),
        np.array([7.1, 6.3, 6.3, 6.4, 6.7, 6.5, 6.3, 6.8, 6.3, 0.0, 7.5, 6.3])),
    'NIRCam.F277W': (
        np.array([67.4, 56.2, 82.0, 94.2, 29.3, 46.3, 41.0, 43.5, 39.2, 44.6, 44.9, 27.8]),
        np.full(12, 3.5)),
    'NIRCam.F444W': (
        np.array([92.3, 47.3, 80.3, 142.5, 89.7, 58.4, 47.1, 44.5, 45.7, 27.1, 32.2, 21.1]),
        np.full(12, 3.9)),
}


def flux_limits(band):
    """log10 EVS flux (nJy): confidence intervals, and the f_star = 1 limit."""
    g = np.loadtxt(f"data/flux_grid_{band}_{sps_tag}.txt")
    ci = np.vstack([compute_conf_ints(mstar_pdf[i], log10m, g[:, i]) for i in redshift_idx])
    fs1 = 10**mstar_fs1 * g[0] / 10**log10m[0]  # flux is linear in stellar mass
    return (np.log10(np.where(ci > 0, ci, 1e-30)),
            np.log10(np.where(fs1 > 0, fs1, 1e-30)))


fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True, layout="constrained")

for ax, band in zip(axes, bands):
    CI, F_fs1 = flux_limits(band)
    ax.fill_between(z, CI[:, 0], CI[:, 6], color=colors[0], zorder=1)
    ax.fill_between(z, CI[:, 1], CI[:, 5], color=colors[1], zorder=1)
    ax.fill_between(z, CI[:, 2], CI[:, 4], color=colors[2], zorder=1)
    ax.plot(z, CI[:, 3], linestyle='dotted', c='black', lw=2, zorder=3)
    ax.plot(z, F_fs1, linestyle='dashed', c='black', lw=1.4, zorder=3)

    flux, err = obs_data[band]
    det = (err > 0) & (flux > err)   # S/N > 1
    lim = (err > 0) & (flux <= err)  # otherwise show a 1 sigma upper limit

    ax.errorbar(z_obs[det], np.log10(flux[det]), xerr=zerr[:, det],
                yerr=(np.log10(flux[det]) - np.log10(flux[det] - err[det]),
                      np.log10(flux[det] + err[det]) - np.log10(flux[det])),
                fmt='o', c='orange', ms=5, lw=1.2, zorder=5)
    ax.errorbar(z_obs[lim], np.log10(flux[lim] + err[lim]), xerr=zerr[:, lim],
                yerr=0.3, uplims=True, fmt='o', c='orange', ms=5, lw=1.2, zorder=5)

    ax.set_xlim(2, 18)
    ax.set_ylim(-1.5, 8.6)
    ax.set_xlabel('$z$', size=12)
    ax.text(0.15, 0.95, band.split('.')[-1], size=12, color='black',
            va='top', transform=ax.transAxes)

axes[0].set_ylabel(r"$\log_{10}(F_\nu \,/\, \mathrm{nJy})$", size=12)
axes[0].text(0.03, 0.03, r'COSMOS-Web, $0.28\,\mathrm{deg^{2}}$', size=12,
             va='bottom', transform=axes[0].transAxes)

handles = [plt.Line2D([0], [0], color=colors[2], lw=5),
           plt.Line2D([0], [0], color=colors[1], lw=5),
           plt.Line2D([0], [0], color=colors[0], lw=5),
           plt.Line2D([0], [0], color='black', linestyle='dotted', lw=2),
           plt.Line2D([0], [0], color='black', linestyle='dashed', lw=1.4),
           plt.Line2D([0], [0], color='orange', marker='o', linestyle='None', ms=5)]
labels = [r'$1\sigma$', r'$2\sigma$', r'$3\sigma$',
          r'$\mathrm{med}(F_{\nu}^{\mathrm{max}})$',
          r'$f_{\star} = 1$; $+3\sigma$', 'Casey+24']
axes[0].legend(handles=handles, labels=labels, frameon=False, loc='upper right',
               markerfirst=False, alignment='right', fontsize=11)

plt.savefig(f'plots/evs_{_obs_str}.png', bbox_inches='tight', dpi=200)
print(f"wrote plots/evs_{_obs_str}.png")
