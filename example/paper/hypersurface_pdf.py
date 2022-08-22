import numpy as np
from scipy import integrate
from scipy.stats import uniform,norm,expon,truncnorm,lognorm,gaussian_kde
import matplotlib.pyplot as plt

from astropy.cosmology import Planck15
import hmf
from evstats import evs
from evstats.stellar import _trunc_lognormal


mass_function = hmf.MassFunction(hmf_model='Behroozi')
mass_function.cosmo_model = Planck15
mass_function.update(dlog10m = 0.001, Mmin = 6, Mmax = 20)
log10m = np.log10(mass_function.m)[:-1]

V = 100
lims = [0.00135, 0.02275, 0.159, 0.500, 0.841, 0.97725, 0.99865]
colors = ['steelblue','lightskyblue','powderblue']
zeds = np.arange(0, 22, 0.5)

CI_mhalo = [None for z in zeds]
for i, z in enumerate(zeds):
    mass_function.update(z=z)
    pdf = evs.evs_hypersurface_pdf(mf=mass_function, V=V**3)
    intcum = integrate.cumtrapz(pdf, log10m, initial=0.)  
    CI_mhalo[i] = log10m[[np.max(np.where(intcum < lim)) for lim in lims]]


CI_mhalo = np.vstack(CI_mhalo)

## fixed baryon fraction and stellar fraction
f_b = 0.16; f_s = 0.1
CI_mstar = np.log10(10**CI_mhalo * f_b * f_s)

## Upper limit from baryon fraction
CI_baryon = np.log10(10**CI_mhalo * f_b)

## params for truncated log norm
myclip_a = 0; myclip_b = 1; my_mean = 0.2; my_std = 0.1
_a, _b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std

_N = int(1e5)  # number of MC samples
methods = ['normal','uniform','exponential','lognormal']
pretty_methods = ['Truncated Normal ($f_{\star}$)',
                  'Uniform ($f_{\star}$)',
                  'Exponential ($f_{\star}$)',
                   'Truncated Log-\nNormal ($f_{\star}$)']

cmap = plt.get_cmap('Spectral', 4)

fig, ax = plt.subplots(1,1,figsize=(4.8,3.8))
i = 0; z = 10
mass_function.update(z=z)
pdf = evs.evs_hypersurface_pdf(mf=mass_function, V=V**3)
ax.plot(log10m, pdf, label='Halo mass PDF', linestyle='dashed', color='grey')
ax.plot(np.log10(10**log10m * f_b), pdf, label='Fixed stellar fraction \n($f_{\star} = 1$)', color='black', lw=2)

cumpdf = np.cumsum(pdf) / np.cumsum(pdf)[-1]
randv = np.random.uniform(size=_N)
idx1 = np.searchsorted(cumpdf, randv); idx0 = np.where(idx1==0, 0, idx1-1)
idx1[idx0==0] = 1  # force first index if at edge of domain
frac1 = (randv - cumpdf[idx0]) / (cumpdf[idx1] - cumpdf[idx0])
randdist = log10m[idx0]*(1-frac1) + log10m[idx1]*frac1  # random samples from halo EVS PDF
 
for _c,(method,_pretty) in enumerate(zip(methods, pretty_methods)):
    if method == 'normal': f_s = truncnorm.rvs(_a, _b, loc = my_mean, scale = my_std, size=_N)
    if method == 'uniform': f_s = uniform.rvs(size=_N, loc=0, scale=1)  # U[0,1]
    if method == 'exponential': f_s = expon.rvs(size=_N, scale=0.1)
    if method == 'lognormal': f_s = _trunc_lognormal(-2, 1, N=_N)

    mstar_samp = np.log10(10**randdist * f_s * f_b) # samples from mstar
    kernel = gaussian_kde(mstar_samp, bw_method=0.04)
    pdf = kernel.pdf(log10m)
    ax.plot(log10m, pdf, label=_pretty, lw=2, color=cmap(_c))


ax.legend(frameon=False)
ax.set_xlim(7,12)
ax.set_xlabel('$\mathrm{log_{10}}(M\,/\, M_{\odot})$')
ax.set_ylabel('$\Phi(M^{\mathrm{max}})$')
ax.text(0.05, 0.3, '$V = (100 \, \mathrm{Mpc})^3$', transform=ax.transAxes, alpha=0.7)
ax.text(0.05, 0.22, '$z = 10$', transform=ax.transAxes, alpha=0.7)
plt.show()
# plt.savefig('plots/hypersurface_pdf_comparison.pdf', bbox_inches='tight'); plt.close()