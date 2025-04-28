import numpy as np
import h5py
import matplotlib.pyplot as plt

from astropy.cosmology import Planck15
import astropy.units as u

from evstats import evs
from evstats.stats import compute_conf_ints, eddington_bias
from evstats.stellar import apply_fs_distribution



with h5py.File('../data/evs_data', 'r') as hf:
    log10m = hf['log10m'][:]
    f = hf['f'][:]
    F = hf['F'][:]
    N = hf['N'][:]
    z = hf['z'][:]


_obs_str = 'Casey24_EazY'
whole_sky = (41252.96 * u.deg**2).to(u.arcmin**2)
survey_area = 0.28 * u.arcmin**2
fsky = float(survey_area / whole_sky)

phi_max = evs._apply_fsky(N, f, F, fsky)

f_b = 0.156
CI_mhalo = compute_conf_ints(phi_max, log10m)
CI_baryon = np.log10(10**CI_mhalo * f_b)

mstar_pdf = np.vstack([apply_fs_distribution(_phi_max, log10m, _N=int(1e5), method='lognormal')
                       for _phi_max in phi_max])
CIs = compute_conf_ints(mstar_pdf, log10m)

# Convert all mass quantities to luminosities
CIs_lum = CIs + 18.78
CI_baryon_lum = CI_baryon + 18.78


# Plot setup
fig, ax = plt.subplots(1, 1, figsize=(5, 5))

low_z_colors = ['steelblue', 'lightskyblue', 'powderblue']
colors = low_z_colors

ax.fill_between(z, CIs_lum[:, 0], CIs_lum[:, 6], alpha=1, color=colors[0])
ax.fill_between(z, CIs_lum[:, 1], CIs_lum[:, 5], alpha=1, color=colors[1])
ax.fill_between(z, CIs_lum[:, 2], CIs_lum[:, 4], alpha=1, color=colors[2])
ax.plot(z, CIs_lum[:, 3], linestyle='dotted', c='black')
ax.plot(z, CI_baryon_lum[:, 6], linestyle='dashed', color='black')

# Observed points
z_obs = np.array([10.47, 12.47, 12.10, 12.16, 9.92, 10.49, 11.19, 11.73, 11.88, 12.6, 12.7, 16.9])
zerr = np.array([[0.20, 0.02, 0.08, 0.27, 0.26, 0.22, 0.34, 0.18, 0.02, 2.5, 4.7, 1.6],
                 [0.42, 0.27, 0.26, 0.23, 0.30, 0.28, 0.19, 0.23, 0.54, 0.1, 0.1, 2.4]])

M = np.log10(np.array([3.7, 4.0, 4.8, 5.7, 11.0, 1.8, 1.9, 1.7, 1.6, 0.59, 5.6, 50.51]) * 10**9)
M_err = np.array([[0.20265029, 0.13033377, 0.32308014, 0.11335686, 0.02679318, 0.1426675,
                   0.1836444, 0.18452443, 0.19382003, 0.9765598, 0.20605448, 0.81225782],
                  [0.13936845, 0.12221588, 0.17001711, 0.09538349, 0.01168576, 0.08715018,
                   0.08297424, 0.13127891, 0.11809931, 0.65890027, 0.14390658, 0.6560418]])

M_corr, _epsilon = eddington_bias(np.log10(10**M * (1. / f_b)), M_err)  # Probably not included
M_corr = np.log10(10**M_corr * f_b)

# Converting observed mass to lum
M_lum = M + 18.78
M_corr_lum = M_corr + 18.78

ax.errorbar(z_obs, M_lum, xerr=zerr, yerr=M_err, fmt='o', c='grey')
ax.errorbar(z_obs, M_corr_lum, xerr=zerr, yerr=M_err, fmt='o', c='orange', label='Casey24_EazY')

ax.set_xlim(7, 18)
ax.set_ylim(23, 30)
ax.set_xlabel('$z$', size=17)
ax.set_ylabel('$\mathrm{log_{10}}(L_{\\nu} \,/\, \mathrm{erg\,s^{-1}\,Hz^{-1}})$', size=15)
ax.text(0.05, 0.04, '$A = 0.28 \; \mathrm{arcmin}^2$', size=12, color='black', alpha=0.8, transform=ax.transAxes)

leg = ax.legend(frameon=False, bbox_to_anchor=(0.44, 0.19), fontsize=12, handletextpad=0.2)
plt.gca().add_artist(leg)

# Custom legend patches
line1 = plt.Line2D((0, 1), (0, 0), color=colors[0], linewidth=5)
line2 = plt.Line2D((0, 1), (0, 0), color=colors[1], linewidth=5)
line3 = plt.Line2D((0, 1), (0, 0), color=colors[2], linewidth=5)
line4 = plt.Line2D((0, 1), (0, 0), color='black', linestyle='dotted', linewidth=2)
line5 = plt.Line2D((0, 1), (0, 0), color='black', linestyle='dashed', linewidth=2)
line_dummy = plt.Line2D((0, 1), (0, 0), color='white')

leg = ax.legend(handles=[line4, line5, line_dummy, line3, line2, line1],
                labels=['$\mathrm{med}(L_{\\nu})$', '$f_{\star} = 1$; $+3\sigma$', '',
                        '$1\sigma$', '$2\sigma$', '$3\sigma$'],
                frameon=False, loc='upper right', fontsize=12, ncol=2)

vp = leg._legend_box._children[-1]._children[0]
for c in vp._children:
    c._children.reverse()
vp.align = "right"

plt.show()
plt.savefig('plots/evs_%s_luminosity.pdf' % _obs_str, bbox_inches='tight', dpi=200)