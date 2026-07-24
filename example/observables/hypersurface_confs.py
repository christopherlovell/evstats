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

_obs_str = f'confs'

whole_sky = (41252.96 * u.deg**2).to(u.arcmin**2)
survey_area = 0.28 * u.degree**2
fsky = float(survey_area / whole_sky)
phi_max = evs._apply_fsky(N, f, F, fsky)

# Confidence interval
redshift_idx = np.arange(len(z))

# SPS grid files for the selected Double Power Law SFH
sfh_tag = "DoublePowerLaw_peak_age0.2_alpha1_beta-1"

sps_grids = {
    "BPASS":f"{sfh_tag}_bpass-2.2.1-bin_chabrier03-0.1,100.0",
}

#z of BAGPIPES
z_obs = np.array([9.69, 12.08, 11.46, 11.46, 9.15, 9.82, 10.20, 11.19, 10.63, 13.2, 13.4, 14.0])
zerr = np.array([[0.25,0.16,0.04,0.23,0.35,0.45,0.54,0.3,0.46,0.9,1.2,2.4],
                [0.24,0.13,0.43,0.28,0.29,0.22,0.51,0.31,0.52,0.6,0.7,1.1]])
 
# # F115W (" Aperture-Based based photometry") - Data commented out, only model lines plotted
# flux_F115W = np.array([-8.4, -4.9, 1.2, -0.3, -2.0, -0.3, -4.5, 9.2, -2.4, -4.7, -8.0, 2.9])
# flux_err_F115W = np.array([8.6, 7.6, 7.8, 7.9, 8.2, 7.7, 8.0, 8.5, 7.6, 8.0, 9.7, 7.6])

# F150W (" Aperture-Based based photometry")
flux_F150W = np.array([60.1, 12.1, 26.2, 18.7, 22.5, 37.1, 24.3, 20.6, 20.8, 0.0, 2.4, 3.4])
flux_err_F150W = np.array([7.1, 6.3, 6.3, 6.4, 6.7, 6.5, 6.3, 6.8, 6.3, 0.0, 7.5, 6.3])

# F277W (" Aperture-Based based photometry")
flux_F277W = np.array([67.4, 56.2, 82.0, 94.2, 29.3, 46.3, 41.0, 43.5, 39.2, 44.6, 44.9, 27.8])
flux_err_F277W = np.array([3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5])

# F444W (" Aperture-Based based photometry")
flux_F444W = np.array([92.3, 47.3, 80.3, 142.5, 89.7, 58.4, 47.1, 44.5, 45.7, 27.1, 32.2, 21.1])
flux_err_F444W = np.array([3.9, 3.9, 3.9, 3.9, 3.9, 3.9, 3.9, 3.9, 3.9, 3.9, 3.9, 3.9])

obs_data = {
    # 'NIRCam.F115W': (flux_F115W, flux_err_F115W),  # No observations plotted for F115W
    'NIRCam.F150W': (flux_F150W, flux_err_F150W),
    'NIRCam.F277W': (flux_F277W, flux_err_F277W),
    'NIRCam.F444W': (flux_F444W, flux_err_F444W),
}


fig, axes = plt.subplots(2, 2, figsize=(24, 20))
axes = axes.flatten()


low_z_colors = ['steelblue','lightskyblue','powderblue'] # ['brown','lightcoral','mistyrose']
colors = low_z_colors

# for ax, band in zip(axes, ['NIRCam.F115W', 'NIRCam.F277W' , 'NIRCam.F444W', 'MIRI.F770W']):
for ax, band in zip(axes, ['NIRCam.F115W', 'NIRCam.F150W' , 'NIRCam.F277W', 'NIRCam.F444W']):


    CI_flux = {}
    for sps, tag in sps_grids.items():
        flux_grid = np.loadtxt(f"data/flux_grid_{band}_{tag}.txt")
        ci_list = []
        for i in redshift_idx:
            ci = compute_conf_ints(phi_max[i], flux_grid[:, i])
            ci_safe = np.where(ci > 0, ci, 1e-30)
            ci_list.append(np.log10(ci_safe))
        CI_flux[sps] = np.vstack(ci_list)

    # Use the computed confidence interval array rather than the raw flux grid.
    if len(CI_flux) == 1:
        CI_flux = next(iter(CI_flux.values()))
    else:
        CI_flux = CI_flux.get('BPASS', next(iter(CI_flux.values())))

    ax.fill_between(z, CI_flux[:,0], CI_flux[:,6], alpha=1, color=colors[0])
    ax.fill_between(z, CI_flux[:,1], CI_flux[:,5], alpha=1, color=colors[1])
    ax.fill_between(z, CI_flux[:,2], CI_flux[:,4], alpha=1, color=colors[2])
    ax.plot(z, CI_flux[:,3], linestyle='dotted', c='black', linewidth=2.5, zorder=5)
    
    ax.set_xlim(2, 18)
    ax.set_ylim(0, 8)
    ax.set_xlabel('Redshift $z$', fontsize=14)
    ax.set_ylabel('log₁₀(Flux [nJy])', fontsize=14)
    ax.set_title(f'{band}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)


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

# Formatting
handles = [
    plt.Line2D([0], [0], color=colors[0], linewidth=8, label='3σ'),
    plt.Line2D([0], [0], color=colors[1], linewidth=8, label='2σ'),
    plt.Line2D([0], [0], color=colors[2], linewidth=8, label='1σ'),
    plt.Line2D([0], [0], color='black', linestyle=':', linewidth=2.5, label='Median(M^{\star}_{\mathrm{max}})'),
    plt.Line2D([0], [0], color='orange', marker='o', linestyle='None', label='Casey+24 candidates'),
]

labels = ['3σ', '2σ', '1σ', 'Median', 'Casey+24 candidates']
ax.legend(handles=handles, labels=labels, frameon=False, loc='upper right', fontsize=11)

plt.tight_layout()
plt.savefig(f'plots/evs_{_obs_str}.png', bbox_inches='tight', dpi=200)
plt.show()
print(f"Plot saved to plots/evs_{_obs_str}.png")