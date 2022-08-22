import sys
import requests

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from astropy.cosmology import Planck15
import hmf

from evstats import evs
from evstats.stats import compute_conf_ints
from evstats.stellar import apply_fs_distribution


_label = sys.argv[1] # EAGLE or FLARES

if _label == 'EAGLE':
    V = 100  # EAGLE box
elif _label == 'FLARES':
    V = 550 # 3200  # FLARES box
else:
    raise ValueError('Simulation not recognised.')


mass_function = hmf.MassFunction(hmf_model='Behroozi')
mass_function.cosmo_model = Planck15
mass_function.update(dlog10m = 0.001, Mmin = 6, Mmax = 20)
log10m = np.log10(mass_function.m)[:-1]

lims = [0.00135, 0.02275, 0.159, 0.500, 0.841, 0.97725, 0.99865]
colors = ['steelblue','lightskyblue','powderblue']
zeds = np.arange(0, 22, 0.5)

_N = int(1e5)  # number of MC samples
f_b = 0.16

CI_mhalo = [None for z in zeds]
CI_mstar = [None for z in zeds]
for i, z in enumerate(zeds):
    mass_function.update(z=z)
    pdf = evs.evs_hypersurface_pdf(mf=mass_function, V=V**3)

    ## halo mass CIs
    CI_mhalo[i] = compute_conf_ints(pdf, log10m)
    intcum = integrate.cumtrapz(pdf, log10m, initial=0.)
    CI_mhalo[i] = log10m[[np.max(np.where(intcum < lim)) for lim in lims]]
    
    mstar_pdf = apply_fs_distribution(pdf, log10m, method='lognormal', _N=_N, f_b=f_b)
    CI_mstar[i] = compute_conf_ints(mstar_pdf, log10m)


CI_mhalo = np.vstack(CI_mhalo)
CI_mstar = np.vstack(CI_mstar)

## Upper limit from baryon fraction
CI_baryon = np.log10(10**CI_mhalo * f_b)

## halo plot
colors = ['peru','sandybrown','peachpuff']
fig, ax = plt.subplots(1,1, figsize=(4.5,3.8))
ax.fill_between(zeds, CI_mhalo[:,0], CI_mhalo[:,6], alpha=1, color=colors[0])
ax.fill_between(zeds, CI_mhalo[:,1], CI_mhalo[:,5], alpha=1, color=colors[1])
ax.fill_between(zeds, CI_mhalo[:,2], CI_mhalo[:,4], alpha=1, color=colors[2])
_med_line = ax.plot(zeds, CI_mhalo[:,3], linestyle='dashed', c='black', lw=0.5)

## load EAGLE data (see download_eagle.py)
if _label == 'EAGLE': _zeds,m_max = np.loadtxt('data/eagle_m200_max.txt')
if _label == 'FLARES': _zeds,m_max = np.loadtxt('data/flares_m200_max.txt')
_sim_line = ax.scatter(_zeds, m_max, marker='o', s=10, color='red', label=_label)

if _label == 'EAGLE': 
    ax.set_xlim(0,11)#21)
    ax.set_ylim(10,)#21)
if _label == 'FLARES': 
    ax.set_xlim(4.5,15.5)#21)
    ax.set_ylim(10,14)#21)
    
    
line1 = plt.Line2D((0,1),(0,0), color=colors[0], linewidth=5)
line2 = plt.Line2D((0,1),(0,0), color=colors[1], linewidth=5)
line3 = plt.Line2D((0,1),(0,0), color=colors[2], linewidth=5)
line_dummy = plt.Line2D((0,1),(0,0), color='white')
leg = ax.legend(handles=[_med_line[0], _sim_line, line_dummy, line3, line2, line1],
          labels=['$\mathrm{med}(M_{\mathrm{max}}^{200})$',_label,'','$1\sigma$','$2\sigma$','$3\sigma$'], 
                frameon=False, ncol=2)#, bbox_to_anchor=(0.8,0.85), fontsize=12)

vp = leg._legend_box._children[-1]._children[0] 
for c in vp._children: 
    c._children.reverse() 
vp.align="right" 

ax.set_xlabel('$z$')
ax.set_ylabel('$\mathrm{log_{10}}(M^{\mathrm{200}}_{\mathrm{max}} \,/\, \mathrm{M_{\odot}})$')
ax.text(0.04, 0.08,'$V = (%i \; \mathrm{Mpc})^3$'%V, transform=ax.transAxes, size=11, alpha=0.6)#, ha='right')

# plt.show()
plt.savefig('plots/hypersurface_halo_%s.pdf'%_label, bbox_inches='tight'); plt.close()


## mstar plot
colors = ['steelblue','lightskyblue','powderblue']

fig, ax = plt.subplots(1,1, figsize=(4.5,4.2))
ax.fill_between(zeds, CI_mstar[:,0], CI_mstar[:,6], color=colors[0])
ax.fill_between(zeds, CI_mstar[:,1], CI_mstar[:,5], color=colors[1])
ax.fill_between(zeds, CI_mstar[:,2], CI_mstar[:,4], color=colors[2])
_med_line = ax.plot(zeds, CI_mstar[:,3], linestyle='dashed', c='black', lw=0.5)

## plot baryon upper limits
ax.plot(zeds, CI_baryon[:,6], linestyle='dashed', color='black', label='$f_{\mathrm{b}}$ $+3\sigma$')

if _label == 'EAGLE': _zeds,mstar_max = np.loadtxt('data/eagle_mstar_max.txt')
if _label == 'FLARES': _zeds,mstar_max = np.loadtxt('data/flares_mstar_max.txt')
_sim_line = ax.scatter(_zeds, mstar_max, marker='o', s=10, color='red', label=_label)

if _label == 'EAGLE': 
    ax.set_xlim(0,11)#21)
    ax.set_ylim(7.5,15)#21)
if _label == 'FLARES': 
    ax.set_xlim(4.5,15.5)#21)
    ax.set_ylim(7,13.4)#21)
ax.set_xlabel('$z$', size=12)
ax.set_ylabel('$\mathrm{log_{10}}(M^{\star}_{\mathrm{max}} \,/\, \mathrm{M_{\odot}})$', size=12)

line1 = plt.Line2D((0,1),(0,0), color=colors[0], linewidth=5)
line2 = plt.Line2D((0,1),(0,0), color=colors[1], linewidth=5)
line3 = plt.Line2D((0,1),(0,0), color=colors[2], linewidth=5)
line5 = plt.Line2D((0,1),(0,0), color='black', linestyle='dashed', linewidth=2)
leg = ax.legend(handles=[_med_line[0],line5,_sim_line,line3,line2,line1],
       labels=['$\mathrm{med}(M_{\mathrm{max}}^{\star})$','$f_{\mathrm{s}} = 1$; $+3\sigma$',_label,
               '$1\sigma$','$2\sigma$','$3\sigma$'], 
                frameon=False, loc='upper right', ncol=2)#, bbox_to_anchor=(0.8,0.85), fontsize=12)

vp = leg._legend_box._children[-1]._children[0] 
for c in vp._children: 
    c._children.reverse() 
vp.align="right" 

ax.text(0.04, 0.04,'$V = (%i \; \mathrm{Mpc})^3$'%V, transform=ax.transAxes, size=11, alpha=0.6)#, ha='right')
ax.text(0.04, 0.12, '$f_{\star} = \mathrm{log}(\mathcal{N}(\mu,\,\sigma^{2}))$', 
        transform=ax.transAxes, size=11, alpha=0.6)#, ha='right')

# plt.show()
plt.savefig('plots/hypersurface_mstar_lognorm_%s.pdf'%_label, bbox_inches='tight'); plt.close()