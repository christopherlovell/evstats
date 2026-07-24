import numpy as np
import h5py
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.cosmology import Planck15

from evstats import evs
from evstats.stats import compute_conf_ints


with h5py.File('../data/evs_all.h5', 'r') as hf:
    log10m = hf['log10m'][:]
    f = hf['f'][:]
    F = hf['F'][:]
    N = hf['N'][:]
    z = hf['z'][:]


whole_sky = (41252.96 * u.deg**2).to(u.arcmin**2)
survey_area = 0.28 * u.deg**2
fsky = float(survey_area / whole_sky)
phi_max = evs._apply_fsky(N, f, F, fsky)
redshift_idx = np.arange(len(z))
_obs_str = f'sps_double_powerlaw_F444W'

# SPS grid files (just copy and paste)
sfh_tag = "DoublePowerLaw_peak_age0.2_alpha1_beta-1"

sps_grids = {
    "BC03":  f"{sfh_tag}_bc03-2016-Miles_chabrier-0.1,100",
    "FSPS": f"{sfh_tag}_fsps-3.2-mistmiles_chabrier03-0.5,120",
    "BPASS":f"{sfh_tag}_bpass-2.2.1-bin_chabrier03-0.1,100.0",
}


#z of BAGPIPES
z_obs = np.array([9.69, 12.08, 11.46, 11.46, 9.15, 9.82, 10.20, 11.19, 10.63, 13.2, 13.4, 14.0])
zerr = np.array([[0.25,0.16,0.04,0.23,0.35,0.45,0.54,0.3,0.46,0.9,1.2,2.4],
                [0.24,0.13,0.43,0.28,0.29,0.22,0.51,0.31,0.52,0.6,0.7,1.1]])
 
# F115W (" Aperture-Based based photometry")
flux_F115W = np.array([-8.4, -4.9, 1.2, -0.3, -2.0, -0.3, -4.5, 9.2, -2.4, -4.7, -8.0, 2.9])
flux_err_F115W = np.array([8.6, 7.6, 7.8, 7.9, 8.2, 7.7, 8.0, 8.5, 7.6, 8.0, 9.7, 7.6])

# F150W (" Aperture-Based based photometry")
flux_F150W = np.array([60.1, 12.1, 26.2, 18.7, 22.5, 37.1, 24.3, 20.6, 20.8, 0.0, 2.4, 3.4])
flux_err_F150W = np.array([7.1, 6.3, 6.3, 6.4, 6.7, 6.5, 6.3, 6.8, 6.3, 7.3, 7.5, 6.3])

# F277W (" Aperture-Based based photometry")
flux_F277W = np.array([67.4, 56.2, 82.0, 94.2, 29.3, 46.3, 41.0, 43.5, 39.2, 44.6, 44.9, 27.8])
flux_err_F277W = np.array([3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5])

# F444W (" Aperture-Based based photometry")
flux_F444W = np.array([92.3, 47.3, 80.3, 142.5, 89.7, 58.4, 47.1, 44.5, 45.7, 27.1, 32.2, 21.1])
flux_err_F444W = np.array([3.9, 3.9, 3.9, 3.9, 3.9, 3.9, 3.9, 3.9, 3.9, 3.9, 3.9, 3.9])

obs_data = {
    'NIRCam.F115W': (flux_F115W, flux_err_F115W),
    'NIRCam.F150W': (flux_F150W, flux_err_F150W),
    'NIRCam.F277W': (flux_F277W, flux_err_F277W),
    'NIRCam.F444W': (flux_F444W, flux_err_F444W),
}


# Plotting - F444W only
fig, ax = plt.subplots(figsize=(8, 8))

band = 'NIRCam.F444W'

# Loads the grids for each SPS model
CI_flux = {}

for sps, tag in sps_grids.items():
    flux_grid = np.loadtxt(f"data/flux_grid_{band}_{tag}.txt")

    ci_list = []
    for i in redshift_idx:
        ci = compute_conf_ints(phi_max[i], flux_grid[:, i])
        ci_safe = np.where(ci > 0, ci, 1e-30)
        ci_list.append(np.log10(ci_safe))

    CI_flux[sps] = np.vstack(ci_list)

# SPS Models
ax.plot(z, CI_flux["BC03"][:, 3],
        linestyle='--', color='coral', linewidth=2.5,
        label='BC03')

ax.plot(z, CI_flux["FSPS"][:, 3],
        linestyle='-', color='steelblue', linewidth=2,
        label='FSPS')

ax.plot(z, CI_flux["BPASS"][:, 3],
        linestyle='-.', color='mediumseagreen', linewidth=2,
        label='BPASS')

# Observed JWST sources
if band in obs_data:
    flux_obs, flux_err_obs = obs_data[band]
    mask = flux_obs > 0

    ax.errorbar(
        z_obs[mask],
        np.log10(flux_obs[mask]),
        xerr=zerr[:, mask],
        yerr=(
            np.log10(flux_obs[mask]) -
            np.log10(flux_obs[mask] - flux_err_obs[mask]),
            np.log10(flux_obs[mask] + flux_err_obs[mask]) -
            np.log10(flux_obs[mask]),
        ),
        fmt='o',
        color='orange',
        markersize=5,
        capsize=3,
        zorder=10,
    )

ax.set_xlim(2, 18)
ax.set_ylim(0, 8)
ax.set_xlabel('$z$', fontsize=14)
ax.set_ylabel("log10(Flux [nJy])", size=17)
ax.text(0.05, 0.95, band, transform=ax.transAxes, verticalalignment='top', fontsize=12)

# Formatting
handles = [
    plt.Line2D([0], [0], color='coral', linestyle='--', linewidth=2.5),
    plt.Line2D([0], [0], color='steelblue', linestyle='-', linewidth=2.5),
    plt.Line2D([0], [0], color='mediumseagreen', linestyle='-.', linewidth=2.5),
    plt.Line2D([0], [0], color='orange', marker='o', linestyle='None'),
]

labels = ['BC03', 'FSPS', 'BPASS', 'JWST candidates']
ax.legend(handles=handles, labels=labels, frameon=False, loc='upper right', fontsize=12, ncol=2)

plt.savefig(f'plots/evs_{_obs_str}_F444W.png', bbox_inches='tight', dpi=200)
plt.show()
