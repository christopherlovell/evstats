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


# Load flux grid data
band = 'NIRCam.F444W'


_obs_str = f'tau_{band}'

whole_sky = (41252.96 * u.deg**2).to(u.arcmin**2)
survey_area = 0.28 * u.degree**2
fsky = float(survey_area / whole_sky)
phi_max = evs._apply_fsky(N, f, F, fsky)
f_b = 0.156

CI_mhalo = compute_conf_ints(phi_max, log10m)
CI_baryon = np.log10(10**CI_mhalo * f_b) 


# mstar_pdf = np.vstack([
#     apply_fs_distribution(
#         _phi_max,
#         log10m,
#         _N=int(1e5),
#         method='lognormal'
#     ) for _phi_max in phi_max
# ])


# Confidence interval
redshift_idx = np.arange(len(z))
# CI_flux = np.log10(np.vstack([compute_conf_ints(mstar_pdf[i], flux_grid[:, i]) for i in redshift_idx]))
 
# low_z_colors = ['steelblue','lightskyblue','powderblue'] # ['brown','lightcoral','mistyrose']
# colors = low_z_colors

fig, ax = plt.subplots(1, 1, figsize=(5,5))

import matplotlib as mpl
cmap = mpl.colormaps['viridis']
colors = cmap(np.linspace(0, 1, 5))

for i, tau in enumerate([-20, -50, -100, -500, -1000]): 
    flux_grid = np.loadtxt(f"data/flux_grid_{band}_tau_m{-1 * tau}.txt")
    CI_flux = np.log10(np.vstack([compute_conf_ints(phi_max[i], flux_grid[:, i]) for i in redshift_idx]))

    # ax.fill_between(z, CI_flux[:,0], CI_flux[:,6], alpha=1, color=colors[0])
    # ax.fill_between(z, CI_flux[:,1], CI_flux[:,5], alpha=1, color=colors[1])
    # ax.fill_between(z, CI_flux[:,2], CI_flux[:,4], alpha=1, color=colors[2])
    ax.plot(z, CI_flux[:,5], label=tau, color=colors[i])
    # ax.plot(z, CI_baryon[:,6], linestyle='dashed', color='black')


#z of EazY
z_obs = np.array([10.47, 12.47, 12.10, 12.16, 9.92, 10.49, 11.19, 11.73, 11.88, 12.6, 12.7, 16.1])
zerr = np.array([[0.42, 0.27, 0.26, 0.23, 0.30, 0.28, 0.19, 0.23, 0.54, 0.1, 0.1, 1.6],
                [0.20, 0.02, 0.08, 0.27, 0.26, 0.22, 0.34, 0.18, 0.02, 2.5, 4.7, 2.4]])


# F444W ("SE++ model based photometry")
flux = np.array([246.0, 226.2, 188.2, 364.2, 260.5, 139.1, 132.9, 106.6, 106.4, 48.7, 41.3, 30.2, 86.0, 33.5, 61.5])[:12]
flux_err = np.array([8.1, 7.1, 7.3, 8.5, 7.7, 8.6, 6.9, 6.8, 5.1, 6.3, 5.8, 4.8, 8.8, 5.7, 5.1])[:12]

#F277W flux values
# F = np.array([67.4, 56.2, 82.0, 94.2, 29.3, 46.3, 41.0, 43.5, 29.2, 44.6, 44.9, 27.8]) # Maybe need np.log F and F_err?
# F_err = np.full((2, 12), 3.5)

# M_corr, _epsilon = eddington_bias(np.log10(10**M * (1./f_b)), M_err)
# M_corr = np.log10(10**M_corr * f_b) 

ax.errorbar(
    z_obs,
    np.log10(flux),
    xerr=zerr,
    yerr=(
        np.log10(flux) - np.log10(flux - flux_err),
        np.log10(flux + flux_err) - np.log10(flux),
    ),
    fmt='o',
    c='orange'
)

# ax.errorbar(z_obs, M_corr, xerr=zerr, yerr=M_err, fmt='o', c='orange', label='Casey24')

ax.set_xlim(2, 18)
ax.set_ylim(1,9)
ax.set_xlabel('$z$', size=17)
ax.set_ylabel("Flux [nJy]", size = 17)
#ax.set_ylabel('$\mathrm{log_{10}}(M^{\star}_{\mathrm{max}} \,/\, M_{\odot})$', size=15)
ax.text(0.05, 0.04, r'$A = 0.28 \; \mathrm{arcmin}^2$', size=12, color='black', alpha=0.8, transform = ax.transAxes)

leg = ax.legend(frameon=False, fontsize=12, handletextpad=0.2, title=r'$\tau$') 
# plt.gca().add_artist(leg) # Add the legend manually to the current Axes.
# 
# line1 = plt.Line2D((0,1),(0,0), color=colors[0], linewidth=5)
# line2 = plt.Line2D((0,1),(0,0), color=colors[1], linewidth=5)
# line3 = plt.Line2D((0,1),(0,0), color=colors[2], linewidth=5)
# line4 = plt.Line2D((0,1),(0,0), color='black', linestyle='dotted', linewidth=2)
# line5 = plt.Line2D((0,1),(0,0), color='black', linestyle='dashed', linewidth=2)
# line_dummy = plt.Line2D((0,1),(0,0), color='white')
# leg = ax.legend(handles=[line4,line5,line_dummy,line3,line2,line1], 
#            labels=[r'$\mathrm{med}(M^{\star}_{\mathrm{max}})$', r'$f_{\star} = 1$; $+3\sigma$','',
#                r'$1\sigma$', r'$2\sigma$', r'$3\sigma$'],
#                 frameon=False, loc='upper right', fontsize=12, ncol=2)
# 
# vp = leg._legend_box._children[-1]._children[0] 
# for c in vp._children: c._children.reverse() 
# vp.align="right" 

plt.show()
# plt.savefig(f'plots/evs_{_obs_str}.png', bbox_inches='tight', dpi=200)
