import numpy as np
import h5py
import matplotlib.pyplot as plt

from astropy.cosmology import Planck15
import astropy.units as u
from scipy import integrate
from evstats import evs
from evstats.stats import compute_conf_ints, eddington_bias
from evstats.stellar import apply_fs_distribution


with h5py.File('../data/evs_all.h5','r') as hf:
    log10m = hf['log10m'][:]
    f = hf['f'][:]
    F = hf['F'][:]
    N = hf['N'][:]
    z = hf['z'][:]

_obs_str = f'SFH_shaded'

whole_sky = (41252.96 * u.deg**2).to(u.arcmin**2)
survey_area = 0.28 * u.degree**2
fsky = float(survey_area / whole_sky)
phi_max = evs._apply_fsky(N, f, F, fsky)
f_b = 0.156

CI_mhalo = compute_conf_ints(phi_max, log10m)
CI_baryon = np.log10(10**CI_mhalo * f_b) 

# Confidence interval
redshift_idx = np.arange(len(z))

low_z_colors = ['steelblue','lightskyblue','powderblue']
colors = low_z_colors

# Change strings for different model names manually
lognormal_model = 'LogNormal_tau0.8_peak_age0.08'
dpl_model = 'DoublePowerLaw_peak_age0.2_alpha1_beta-1'


fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()


# BUG IN BELOW LOOP WHERE TAKING LOG OF 0 RESULTS IN RUNTIME ERROR, COULD FIX AND REMOVE CI_SAFE LINES TO SIMPLIFY CODE?
for ax, band in zip(axes, ['NIRCam.F277W', 'NIRCam.F115W', 'NIRCam.F444W', 'MIRI.F770W']):

    # Loads minimum tau
    flux_grid_min = np.loadtxt(f"data/flux_grid_{band}_Exponential_tau-0.02.txt")
    CI_flux_min_list = []
    for i in redshift_idx:
        ci = compute_conf_ints(phi_max[i], flux_grid_min[:, i])
        ci_safe = np.where(ci > 0, ci, 1e-30)       #bug
        CI_flux_min_list.append(np.log10(ci_safe))
    CI_flux_min = np.vstack(CI_flux_min_list)
    
    # Loads maximum tau
    flux_grid_max = np.loadtxt(f"data/flux_grid_{band}_Exponential_tau-1.0.txt")
    CI_flux_max_list = []
    for i in redshift_idx:
        ci = compute_conf_ints(phi_max[i], flux_grid_max[:, i])
        ci_safe = np.where(ci > 0, ci, 1e-30)
        CI_flux_max_list.append(np.log10(ci_safe))
    CI_flux_max = np.vstack(CI_flux_max_list)
    
    # Bounds of EVS flux for median most extreme parameters?
    ax.fill_between(z, CI_flux_min[:,3], CI_flux_max[:,3], 
                    alpha=0.3, color=colors[1], zorder=1)
    
    # Plot boundary curves
    ax.plot(z, CI_flux_min[:,3], linestyle='-', c=colors[1], 
           linewidth=1.5, alpha=0.5, zorder=2)
    ax.plot(z, CI_flux_max[:,3], linestyle='-', c=colors[1], 
           linewidth=1.5, alpha=0.5, zorder=2)
    
    # Plot LogNormal
    flux_grid_ln = np.loadtxt(f"data/flux_grid_{band}_{lognormal_model}.txt")
    CI_flux_ln_list = []
    for i in redshift_idx:
        ci_ln = compute_conf_ints(phi_max[i], flux_grid_ln[:, i])
        ci_ln_safe = np.where(ci_ln > 0, ci_ln, 1e-30)
        CI_flux_ln_list.append(np.log10(ci_ln_safe))
    CI_flux_ln = np.vstack(CI_flux_ln_list)
    
    ax.plot(z, CI_flux_ln[:,3], linestyle='--', c='coral', linewidth=2.5, zorder=4)
    
    # Plot DPL
    flux_grid_dpl = np.loadtxt(f"data/flux_grid_{band}_{dpl_model}.txt")
    CI_flux_dpl_list = []
    for i in redshift_idx:
        ci_dpl = compute_conf_ints(phi_max[i], flux_grid_dpl[:, i])
        ci_dpl_safe = np.where(ci_dpl > 0, ci_dpl, 1e-30)
        CI_flux_dpl_list.append(np.log10(ci_dpl_safe))
    CI_flux_dpl = np.vstack(CI_flux_dpl_list)
    
    ax.plot(z, CI_flux_dpl[:,3], linestyle='-.', c='mediumseagreen', linewidth=2.5, zorder=4)
    
    ax.set_xlim(2, 18)
    ax.set_ylim(-3, 8)
    ax.set_xlabel('$z$', size=17)
    ax.set_ylabel("log10(Flux [nJy])", size=17)
    ax.text(0.05, 0.04, f'{band}', size=12, color='black', alpha=0.8, transform=ax.transAxes)


# NEED TO FIX BELOW TO SHOW ON CORRECT PANEL?

# #z of EazY
# z_obs = np.array([10.47, 12.47, 12.10, 12.16, 9.92, 10.49, 11.19, 11.73, 11.88, 12.6, 12.7, 16.1])
# zerr = np.array([[0.42, 0.27, 0.26, 0.23, 0.30, 0.28, 0.19, 0.23, 0.54, 0.1, 0.1, 1.6],
#                 [0.20, 0.02, 0.08, 0.27, 0.26, 0.22, 0.34, 0.18, 0.02, 2.5, 4.7, 2.4]])

# # F444W ("SE++ model based photometry")
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


leg = ax.legend(frameon=False, bbox_to_anchor=(0.44,0.19), fontsize=12, handletextpad=0.2) 
plt.gca().add_artist(leg)

line1 = plt.Line2D((0,1),(0,0), color='lightsteelblue', linewidth=5)
line2 = plt.Line2D((0,1),(0,0), color='steelblue', linestyle='--', linewidth=2, alpha=0.7)
line3 = plt.Line2D((0,1),(0,0), color='coral', linestyle='--', linewidth=2)
line4 = plt.Line2D((0,1),(0,0), color='mediumseagreen', linestyle='-.', linewidth=2)
line_dummy = plt.Line2D((0,1),(0,0), color='white')
leg = ax.legend(handles=[line1,line2,line3,line4,line_dummy], 
           labels=['Exp Range', 'Exp bounds (τ=-0.02, -1.0)', 'LogNormal', 'DPL', ''],
                frameon=False, loc='upper right', fontsize=12, ncol=2)

vp = leg._legend_box._children[-1]._children[0] 
for c in vp._children: c._children.reverse() 
vp.align="right"

plt.tight_layout()
plt.savefig(f'plots/evs_{_obs_str}.png', bbox_inches='tight', dpi=200)
plt.show()
