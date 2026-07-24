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
_obs_str = f"imf_double_powerlaw_F444W"

# IMF grid labels (just copy and paste)
sfh_tag = "DoublePowerLaw_peak_age0.2_alpha1_beta-1"

imf_grids = {
    "chabrier": f"{sfh_tag}_bc03-2016-Miles_chabrier-0.1,100",
    "salpeter": f"{sfh_tag}_bc03-2016-Miles_salpeter-0.1,100",
    "kroupa":   f"{sfh_tag}_bc03-2016-Miles_kroupa-0.1,100",
}

#z of BAGPIPES
z_obs = np.array([9.69, 12.08, 11.46, 11.46, 9.15, 9.82, 10.20, 11.19, 10.63, 13.2, 13.4, 14.0])
zerr = np.array([[0.25,0.16,0.04,0.23,0.35,0.45,0.54,0.3,0.46,0.9,1.2,2.4],
                [0.24,0.13,0.43,0.28,0.29,0.22,0.51,0.31,0.52,0.6,0.7,1.1]])

# F115W ("SE++ model based photometry")
flux_F115W = np.array([0.0, 0.0, 21.9, 8.4, 0, 0.3, 0.0, 0.0, 4.3, 2.5, 2.7, 9.3, 10.7, 0.0, 0.0])[:12]
flux_err_F115W = np.array([9.0, 8.6, 14.4, 11.2, 10.3, 15.1, 10.3, 10.8, 10.0, 11.8, 10.8, 8.9, 11.5, 10.2, 7.6])[:12]

# F150W ("SE++ model based photometry")
flux_F150W = np.array([147.8, 44.1, 51.8, 39.8, 86.8, 89.7, 61.8, 44.5, 43.5, 5.3, 0.0, 1.2, 17.9, 10.0, 2.5])[:12]
flux_err_F150W = np.array([14.3, 10.9, 11.0, 9.2, 11.5, 12.5, 10.2, 10.9, 8.2,  9.8, 8.3, 7.3, 9.4,8.7, 7.1])[:12]

# F277W ("SE++ model based photometry")
flux_F277W = np.array([180.9, 248.2, 182.4, 235.8, 97.4, 110.7, 105.6, 103.7, 86.8, 82.9, 66.9, 50.9, 174.6, 65.8, 77.4])[:12]
flux_err_F277W = np.array([6.9, 6.0, 6.0, 6.3, 5.9, 6.9, 6.0, 5.8, 4.4, 5.9, 5.0, 4.5, 12.8, 5.3, 4.7])[:12]

# F444W ("SE++ model based photometry")
flux_F444W = np.array([246.0, 226.2, 188.2, 364.2, 260.5, 139.1, 132.9, 106.6, 106.4, 48.7, 41.3, 30.2, 86.0, 33.5, 61.5])[:12]
flux_err_F444W = np.array([8.1, 7.1, 7.3, 8.5, 7.7, 8.6, 6.9, 6.8, 5.1, 6.3, 5.8, 4.8, 8.8, 5.7, 5.1])[:12]

obs_data = {
    'NIRCam.F115W': (flux_F115W, flux_err_F115W),
    'NIRCam.F150W': (flux_F150W, flux_err_F150W),
    'NIRCam.F277W': (flux_F277W, flux_err_F277W),
    'NIRCam.F444W': (flux_F444W, flux_err_F444W),
}

# Plotting F444W only, can change band to plot other filters.
fig, ax = plt.subplots(figsize=(8, 8))

band = 'NIRCam.F444W'

# Loads the grids for each IMF
CI_flux = {}

for imf, tag in imf_grids.items():
    flux_grid = np.loadtxt(f"data/flux_grid_{band}_{tag}.txt")

    ci_list = []
    for i in redshift_idx:
        ci = compute_conf_ints(phi_max[i], flux_grid[:, i])
        ci_safe = np.where(ci > 0, ci, 1e-30)
        ci_list.append(np.log10(ci_safe))

    CI_flux[imf] = np.vstack(ci_list)



# IMF Models
ax.plot(z, CI_flux["chabrier"][:, 3],
        linestyle='--', color='coral', linewidth=2.5,
        label='Chabrier IMF')

ax.plot(z, CI_flux["salpeter"][:, 3],
        linestyle='--', color='steelblue', linewidth=2,
        label='Salpeter IMF')

ax.plot(z, CI_flux["kroupa"][:, 3],
        linestyle='--', color='mediumseagreen', linewidth=2,
        label='Kroupa IMF')



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
    plt.Line2D([0], [0], color='steelblue', linestyle='--', linewidth=2.5),
    plt.Line2D([0], [0], color='mediumseagreen', linestyle='--', linewidth=2.5),
    plt.Line2D([0], [0], color='orange', marker='o', linestyle='None'),
]

labels = ['Chabrier', 'Salpeter', 'Kroupa', 'JWST candidates']
ax.legend(handles=handles, labels=labels, frameon=False, loc='upper right', fontsize=12, ncol=2)

plt.savefig(f'plots/evs_{_obs_str}_F444W.png', bbox_inches='tight', dpi=200)
plt.show()
