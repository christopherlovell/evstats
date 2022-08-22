import numpy as np
import h5py
import matplotlib.pyplot as plt
import astropy.units as u

from evstats import evs
from evstats.stats import compute_conf_ints, eddington_bias
from evstats.stellar import apply_fs_distribution


with h5py.File('../data/evs_all.h5','r') as hf:
    log10m = hf['log10m'][:]
    f = hf['f'][:]
    F = hf['F'][:]
    N = hf['N'][:]
    z = hf['z'][:]


_obs_str = 'labbe22'
whole_sky = (41252.96 * u.deg**2).to(u.arcmin**2)
survey_area = 40 * u.arcmin**2
fsky = float(survey_area / whole_sky)

phi_max = evs._apply_fsky(N, f, F, fsky)

f_b = 0.16
CI_mhalo = compute_conf_ints(phi_max, log10m)
CI_baryon = np.log10(10**CI_mhalo * f_b)

mstar_pdf = np.vstack([apply_fs_distribution(_phi_max, log10m, _N=int(1e5), method='lognormal') \
        for _phi_max in phi_max])
CIs = compute_conf_ints(mstar_pdf, log10m)



fig, ax = plt.subplots(1, 1, figsize=(5,5))

low_z_colors = ['brown','lightcoral','mistyrose'] # ['steelblue','lightskyblue','powderblue']
colors = low_z_colors

ax.fill_between(z, CIs[:,0], CIs[:,6], alpha=1, color=colors[0])
ax.fill_between(z, CIs[:,1], CIs[:,5], alpha=1, color=colors[1])
ax.fill_between(z, CIs[:,2], CIs[:,4], alpha=1, color=colors[2])
ax.plot(z, CIs[:,3], linestyle='dotted', c='black')
ax.plot(z, CI_baryon[:,6], linestyle='dashed', color='black')

z_obs = np.array([9.92, 7.56, 8.91, 9.97, 10.83, 9.35, 8.49])
zerr = np.array([[0.],
                 [0.]])

M = np.array([10.93, 11.16, 10.83, 10.76, 10.44, 10.41, 10.18])
M_err = np.array([[0.],
                 [0.]])

M_corr, _epsilon = eddington_bias(np.log10(10**M * (1./f_b)), M_err)
M_corr = np.log10(10**M_corr * f_b)

# plt.errorbar(z_obs, M, xerr=[zmin,zmax], yerr=Merr, fmt='o', c='orange', label='Caputi+15')
ax.errorbar(z_obs, M, xerr=zerr, yerr=M_err, fmt='o', c='grey')
ax.errorbar(z_obs, M_corr, xerr=zerr, yerr=M_err, fmt='o', c='dodgerblue', label='Labbe+22')
#ax.errorbar(z_obs[2:], M_corr[2:], xerr=zerr[:,2:], yerr=M_err[:,2:], fmt='o', c='darkorange')

# z = 17 solution
z_obs = np.array([16, 4.8, 4.9])
zerr = np.array([[0.6, 0.1, 0],
                 [0.6, 0.1, 0.02]])

M = np.array([9.6, 9.6, 8.7])
M_err = np.array([[0.2, 0.2, 0.1],
                 [0.2, 0.5, 0.1]])

M_corr, _epsilon = eddington_bias(np.log10(10**M * (1./f_b)), M_err)
M_corr = np.log10(10**M_corr * f_b)

# plt.errorbar(z_obs, M, xerr=[zmin,zmax], yerr=Merr, fmt='o', c='orange', label='Caputi+15')
ax.errorbar(z_obs, M, xerr=zerr, yerr=M_err, fmt='o', c='grey')
ax.errorbar(z_obs, M_corr, xerr=zerr, yerr=M_err, fmt='o', c='darkblue', label='Naidu+22')



ax.set_xlim(6, 18); ax.set_ylim(6, 12.8)
ax.set_xlabel('$z$', size=17)
ax.set_ylabel('$\mathrm{log_{10}}(M^{\star}_{\mathrm{max}} \,/\, M_{\odot})$', size=15)
ax.text(0.05, 0.04, '$A = 40 \; \mathrm{arcmin}^2$', size=12, color='black', alpha=0.8, transform = ax.transAxes)

leg = ax.legend(frameon=False, bbox_to_anchor=(0.37,0.25), fontsize=12, handletextpad=0.2) 
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
# plt.savefig('plots/evs_%s.pdf'%_obs_str, bbox_inches='tight', dpi=200)
