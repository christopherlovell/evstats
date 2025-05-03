import astropy.units as u
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform,norm,expon,truncnorm,lognorm,gaussian_kde

from evstats import evs
from evstats.stellar import apply_fs_distribution, halo_dependent_fs
from evstats.stats import compute_conf_ints

f_b = 0.16
log10m = np.random.uniform(low=10, high=16, size=(int(4e6),)) 
mstar, f_s = halo_dependent_fs(log10m, lo_halo_mass=13, hi_halo_mass=14)

halo_masses = np.log10(10**mstar / f_s / f_b)

halo_bins = np.linspace(8, 16, 500)

CIs = [None] * (len(halo_bins) - 1)
for i, halom in enumerate(halo_bins[:-1]):

    mask = (halo_masses > halom) & (halo_masses < halo_bins[i+1])

    if np.sum(mask) == 0:
        continue

    f_s_arr = np.logspace(-3, 0, 200)

    kernel = gaussian_kde(f_s[mask], bw_method=0.08)
    pdf = kernel.pdf(f_s_arr)

    CIs[i] = compute_conf_ints(pdf, f_s_arr)
    



CI_mask = np.zeros(len(CIs), dtype=bool)

for i, _CIs in enumerate(CIs):
    if _CIs is not None:
        CI_mask[i] = True

CIs = np.vstack([_CIs for _CIs in CIs if _CIs is not None])

fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))
colors = ['powderblue', 'lightskyblue', 'steelblue']

ax.fill_between(halo_bins[:-1][CI_mask], np.log10(CIs[:,0]), np.log10(CIs[:,6]), color=colors[0])
ax.fill_between(halo_bins[:-1][CI_mask], np.log10(CIs[:,1]), np.log10(CIs[:,5]), color=colors[1])
ax.fill_between(halo_bins[:-1][CI_mask], np.log10(CIs[:,2]), np.log10(CIs[:,4]), color=colors[2])

ax.set_xlabel(r'${\rm log_{10}} \; M_{\rm halo} \,/\, {\rm M_{\odot}}$')
ax.set_ylabel(r'${\rm log_{10}} \; f_{\star}$')

ax.set_xlim(10, 16)
# plt.show()
plt.savefig('plots/f_s_halo_distribution.pdf', bbox_inches='tight', dpi=200)
plt.close()
