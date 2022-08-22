import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cmasher as cmr

from astropy.cosmology import Planck15
import hmf

from evstats import evs

_N = 6
cmap = plt.get_cmap('RdYlBu_r', _N)

mass_function = hmf.MassFunction(hmf_model='Behroozi')
mass_function.cosmo_model = Planck15
mass_function.update(dlog10m = 0.01, Mmin = 7, Mmax = 16)

zeds = np.array([0,2,4,8,12,16])

fig, ax = plt.subplots(1, 1, figsize=(4.5,3.8))
ax2 = ax.twinx()

for i, z in enumerate(zeds):
    mass_function.update(z=z)
    ax.plot(np.log10(mass_function.m), np.log10(mass_function.dndlog10m), 
               label=r'$z = %s$'%z, linewidth=3, c=cmap(i))
    

for i, z in enumerate(zeds):
    mass_function.update(z=z)
    ax2.plot(np.log10(mass_function.m[:-1]), evs.evs_hypersurface_pdf(mf=mass_function, V=100**3), 
             c=cmap(i), linestyle='dotted', linewidth=1.5)
    
    
ax.set_ylim(-7,2); ax.set_xlim(8,16)
ax2.set_ylim(0.03,7); ax2.set_xlim(8,16)

ax.set_xlabel('$\mathrm{log_{10}}(M_{200} \, / \, M_{\odot})$', size=11)

ax.set_ylabel('$\phi (M_{200} \,|\, z)$', size=11) # \; (\mathrm{Mpc}^{-3} \; \mathrm{dex}^{-1})
ax2.set_ylabel('$\phi (M^{\mathrm{Max}}_{200} \,|\, z, V)$', size=11) # \; (\mathrm{Mpc}^{-3} \; \mathrm{dex}^{-1})

ax.text(0.2, 0.9, '$V = (100 \, \mathrm{Mpc})^3$', transform=ax.transAxes, alpha=0.7)



cax = fig.add_axes([0.75, 0.48, 0.03, 0.37])
norm = mpl.colors.Normalize(vmin=-0.5, vmax=len(zeds)-0.5)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, cax=cax, ticks=np.arange(len(zeds)), pad=0.1, label='$z$')#, 
cbar.ax.set_yticklabels(zeds) 

plt.show()
# plt.savefig('plots/hmf.pdf', bbox_inches='tight')