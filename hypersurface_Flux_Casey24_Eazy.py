import numpy as np
import h5py
import matplotlib.pyplot as plt

from astropy.cosmology import Planck15
import astropy.units as u
from scipy import integrate
from evstats import evs
from evstats.stats import compute_conf_ints, eddington_bias
from evstats.stellar import apply_fs_distribution


with h5py.File('../data/evs_data','r') as hf:
    log10m = hf['log10m'][:]
    f = hf['f'][:]
    F = hf['F'][:]
    N = hf['N'][:]
    z = hf['z'][:]


# Loading Flux grid data
data = np.load("fluxes_grid_data_3.npz")
flux_grid = data["fluxes_grid"]
masses = data["masses"] #same
redshift = data["redshift"]
masses = np.log10(masses)

_obs_str = 'Casey24_EazY_FLUX'

# Cosmology Values etc.
whole_sky = (41252.96 * u.deg**2).to(u.arcmin**2)
survey_area = 0.28 * u.arcmin**2
fsky = float(survey_area / whole_sky)
phi_max = evs._apply_fsky(N, f, F, fsky)
f_b = 0.156

CI_mhalo = compute_conf_ints(phi_max, log10m)
CI_baryon = np.log10(10**CI_mhalo * f_b) # Probably need to change this too? 


mstar_pdf = np.vstack([apply_fs_distribution(_phi_max, log10m, _N=int(1e5), method='lognormal')   #change N for load time/ smoothness, 3 = quick, 5 = precise
        for _phi_max in phi_max])



# for i in [25,50,75]:

#     plt.plot(log10m, integrate.cumulative_trapezoid(mstar_pdf[i], flux_grid[i], initial=0.))

#     # plt.plot(np.log10(flux_grid[i,:]), mstar_pdf[i], label=redshift[i])
#     CIs = compute_conf_ints(mstar_pdf[i], flux_grid[i,:])
#     print("redshift = ", redshift[i])
#     print(CIs)
# plt.xlabel("log10(flux)")
# plt.ylabel("PDF")
# plt.legend()
# plt.show()

# intcum = 
# print(CIs)

# # Multiplying flux grid by Mstar grid
#flux_pdf = flux_grid * mstar_pdf

# Confidence interval
redshift_idx = np.arange(len(redshift))
CI_flux = np.log10(np.vstack([compute_conf_ints(mstar_pdf[i], flux_grid[i,:]) for i in redshift_idx]))


fig, ax = plt.subplots(1, 1, figsize=(5,5))

low_z_colors = ['steelblue','lightskyblue','powderblue'] # ['brown','lightcoral','mistyrose']
colors = low_z_colors

ax.fill_between(z, CI_flux[:,0], CI_flux[:,6], alpha=1, color=colors[0])
ax.fill_between(z, CI_flux[:,1], CI_flux[:,5], alpha=1, color=colors[1])
ax.fill_between(z, CI_flux[:,2], CI_flux[:,4], alpha=1, color=colors[2])
ax.plot(z, CI_flux[:,3], linestyle='dotted', c='black')
# ax.plot(z, CI_baryon[:,6], linestyle='dashed', color='black')


#z of EazY
z_obs = np.array([10.47, 12.47, 12.10, 12.16, 9.92, 10.49, 11.19, 11.73, 11.88, 12.6, 12.7, 16.1])
zerr = np.array([[0.24, 0.13, 0.43, 0.28, 0.29, 0.22, 0.51, 0.31, 0.52],
                [0.25, 0.16, 0.04, 0.23, 0.35, 0.45, 0.54, 0.30, 0.46]])

#F277W flux values
F = np.array([67.4, 56.2, 82.0, 94.2, 29.3, 46.3, 41.0, 43.5, 29.2, 44.6, 44.9, 27.8]) # Maybe need np.log F and F_err?
F_err = np.full((2, 12), 3.5)



# M_corr, _epsilon = eddington_bias(np.log10(10**M * (1./f_b)), M_err)
# M_corr = np.log10(10**M_corr * f_b) 



# plt.errorbar(z_obs, F, xerr=[zmin,zmax], yerr=Ferr, fmt='o', c='orange', label='Caputi+15')
# ax.errorbar(z_obs, F, xerr=zerr, yerr=F_err, fmt='o', c='grey')
# ax.errorbar(z_obs, M_corr, xerr=zerr, yerr=M_err, fmt='o', c='orange', label='Casey24')

ax.set_xlim(7,18)
#ax.set_ylim(5,12.8)
ax.set_xlabel('$z$', size=17)
ax.set_ylabel("Flux [nJy]", size = 17)
#ax.set_ylabel('$\mathrm{log_{10}}(M^{\star}_{\mathrm{max}} \,/\, M_{\odot})$', size=15)
ax.text(0.05, 0.04, '$A = 0.28 \; \mathrm{arcmin}^2$', size=12, color='black', alpha=0.8, transform = ax.transAxes)

leg = ax.legend(frameon=False, bbox_to_anchor=(0.44,0.19), fontsize=12, handletextpad=0.2) 
plt.gca().add_artist(leg) # Add the legend manually to the current Axes.

line1 = plt.Line2D((0,1),(0,0), color=colors[0], linewidth=5)
line2 = plt.Line2D((0,1),(0,0), color=colors[1], linewidth=5)
line3 = plt.Line2D((0,1),(0,0), color=colors[2], linewidth=5)
line4 = plt.Line2D((0,1),(0,0), color='black', linestyle='dotted', linewidth=2)
line5 = plt.Line2D((0,1),(0,0), color='black', linestyle='dashed', linewidth=2)
line_dummy = plt.Line2D((0,1),(0,0), color='white')
leg = ax.legend(handles=[line4,line5,line_dummy,line3,line2,line1], 
           labels=['$\mathrm{med}(M^{\star}_{\mathrm{max}})$','$f_{\star} = 1$; $+3\sigma$','',
               '$1\sigma$', '$2\sigma$', '$3\sigma$'],
                frameon=False, loc='upper right', fontsize=12, ncol=2)

vp = leg._legend_box._children[-1]._children[0] 
for c in vp._children: c._children.reverse() 
vp.align="right" 

plt.show()
plt.savefig('plots/evs_%s.pdf'%_obs_str, bbox_inches='tight', dpi=200)



