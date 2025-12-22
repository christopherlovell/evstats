import numpy as np
import h5py
import matplotlib.pyplot as plt

from astropy.cosmology import Planck15
import astropy.units as u
from evstats import evs
from evstats.stats import compute_conf_ints


with h5py.File('../data/evs_all.h5','r') as hf:
    log10m = hf['log10m'][:]
    f = hf['f'][:]
    F = hf['F'][:]
    N = hf['N'][:]
    z = hf['z'][:]

_obs_str = f'SFH_comparison'

whole_sky = (41252.96 * u.deg**2).to(u.arcmin**2)
survey_area = 0.28 * u.degree**2
fsky = float(survey_area / whole_sky)
phi_max = evs._apply_fsky(N, f, F, fsky)

# Confidence interval
redshift_index = np.arange(len(z))

# Model traits
models = [

    {'tag': 'Exponential_tau-0.02', 'label': 'Exp τ=-0.02', 'color': '#1f77b4', 'ls': '-'},
    {'tag': 'Exponential_tau-0.05', 'label': 'Exp τ=-0.05', 'color': '#2ca02c', 'ls': '-'},
    {'tag': 'Exponential_tau-0.1', 'label': 'Exp τ=-0.1', 'color': '#ff7f0e', 'ls': '-'},
    {'tag': 'Exponential_tau-0.5', 'label': 'Exp τ=-0.5', 'color': '#d62728', 'ls': '-'},
    {'tag': 'Exponential_tau-1.0', 'label': 'Exp τ=-1.0', 'color': '#9467bd', 'ls': '-'},

    {'tag': 'LogNormal_tau0.3_peak_age0.03', 'label': 'LN τ=0.3, peak=0.03', 'color': '#8c564b', 'ls': '--'},
    {'tag': 'LogNormal_tau0.5_peak_age0.05', 'label': 'LN τ=0.5, peak=0.05', 'color': '#e377c2', 'ls': '--'},
    {'tag': 'LogNormal_tau0.8_peak_age0.08', 'label': 'LN τ=0.8, peak=0.08', 'color': '#7f7f7f', 'ls': '--'},

    {'tag': 'DoublePowerLaw_peak_age0.05_alpha10_beta-10', 'label': 'DPL peak=0.05', 'color': '#bcbd22', 'ls': '-.'},
    {'tag': 'DoublePowerLaw_peak_age0.1_alpha5_beta-5', 'label': 'DPL peak=0.1', 'color': '#17becf', 'ls': '-.'},
    {'tag': 'DoublePowerLaw_peak_age0.2_alpha1_beta-1', 'label': 'DPL peak=0.2', 'color': '#ff9896', 'ls': '-.'},
]

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()

for ax, band in zip(axes, ['NIRCam.F277W', 'NIRCam.F115W', 'NIRCam.F444W', 'MIRI.F770W']):
    
    # Plot each model
    for model in models:
        flux_grid = np.loadtxt(f"data/flux_grid_{band}_{model['tag']}.txt")
        
        CI_flux_list = []
        for i in redshift_index:
            ci = compute_conf_ints(phi_max[i], flux_grid[:, i])
            ci_safe = np.where(ci > 0, ci, 1e-30)  # Avoid log10(0)
            CI_flux_list.append(np.log10(ci_safe))
        CI_flux = np.vstack(CI_flux_list)
        
        ax.plot(z, CI_flux[:,3], 
               label=model['label'],
               color=model['color'],
               linestyle=model['ls'],
               linewidth=2)
    
    ax.set_xlim(2, 18)
    ax.set_ylim(-3, 8)
    ax.set_xlabel('$z$', size=17)
    ax.set_ylabel("Log10(Flux [nJy])", size=17)
    ax.text(0.05, 0.04, f'{band}', size=12, color='black', alpha=0.8, transform=ax.transAxes)
    
    # Legend only on first panel
    if ax == axes[0]:
        ax.legend(frameon=False, fontsize=10, loc='lower right')

# Observations
# z_obs = np.array([10.47, 12.47, 12.10, 12.16, 9.92, 10.49, 11.19, 11.73, 11.88, 12.6, 12.7, 16.1])
# zerr = np.array([[0.42, 0.27, 0.26, 0.23, 0.30, 0.28, 0.19, 0.23, 0.54, 0.1, 0.1, 1.6],
#                 [0.20, 0.02, 0.08, 0.27, 0.26, 0.22, 0.34, 0.18, 0.02, 2.5, 4.7, 2.4]])

# # F444W
# flux = np.array([246.0, 226.2, 188.2, 364.2, 260.5, 139.1, 132.9, 106.6, 106.4, 48.7, 41.3, 30.2, 86.0, 33.5, 61.5])[:12]
# flux_err = np.array([8.1, 7.1, 7.3, 8.5, 7.7, 8.6, 6.9, 6.8, 5.1, 6.3, 5.8, 4.8, 8.8, 5.7, 5.1])[:12]

# ax.errorbar(
#     z_obs,
#     np.log10(flux),
#     xerr=zerr,
#     yerr=(
#         np.log10(flux) - np.log10(flux - flux_err),
#         np.log10(flux + flux_err) - np.log10(flux),
#     ),
#     fmt='o',
#     c='orange'
# )

plt.tight_layout()
plt.savefig(f'plots/evs_{_obs_str}.png', bbox_inches='tight', dpi=200)
# plt.show()  # Comment out if running non-interactively