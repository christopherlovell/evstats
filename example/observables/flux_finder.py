import matplotlib.pyplot as plt
import numpy as np
import h5py
from astropy.cosmology import Planck18 as cosmo
from multiprocessing import Pool
from unyt import Hz, Msun, Myr, erg, nJy, s, angstrom, Gyr

from synthesizer import galaxy
from synthesizer.instruments import FilterCollection
from synthesizer.grid import Grid
from synthesizer.parametric import SFH, Stars, ZDist
from synthesizer.emission_models.attenuation import Madau96
from synthesizer.emission_models import IncidentEmission


# nprocs = 2

with h5py.File('../data/evs_all.h5','r') as hf:
    log10m = hf['log10m'][:]
    f = hf['f'][:]
    F = hf['F'][:]
    N = hf['N'][:]
    redshifts = hf['z'][:]


grid_name = "test_grid"
grid_dir = "../../../synthesizer/tests/test_grid/"
grid = Grid(
    grid_name,
    grid_dir=grid_dir,
    new_lam=np.logspace(2.3, 5, 500) * angstrom
)

model = IncidentEmission(grid=grid)



filter_codes = [
    f"JWST/{filt}" for filt in 
    ['NIRCam.F277W', 'NIRCam.F115W', 'NIRCam.F444W', 'MIRI.F770W']
]
fc = FilterCollection(filter_codes, new_lam=grid.lam)

flux = np.zeros((len(log10m), len(redshifts)))

def create_galaxy(z, m, grid, tau): 
    sfh = SFH.Exponential(tau, cosmo.age(z) * Gyr)
    metal_dist = ZDist.Normal(mean=0.01, sigma=0.005)

    exp_stars = Stars(
        grid.log10age,
        grid.metallicity,
        sf_hist=sfh,
        metal_dist=metal_dist,
        initial_mass=m * Msun,
    )
    
    return galaxy(stars=exp_stars, redshift=z)



import matplotlib as mpl

fig, ax = plt.subplots(1, 1, figsize=(6, 5))
cmap = mpl.colormaps['viridis']
colors = cmap(np.linspace(0, 1, 5))
for i, tau in enumerate([-20, -50, -100, -500, -1000]):
    gal = create_galaxy(1, 10, grid, tau * Myr)
    # ax.step(10**gal.stars.log10ages / 1e9, gal.stars.sf_hist, where="mid", label=f'{tau} Myr')
    ax.step(gal.stars.log10ages, gal.stars.sf_hist, where="mid", label=f'{tau} Myr', color=colors[i])
    # gal.stars.plot_sfh(ax=ax)

# ax.set_xlim(0, 10)
ax.set_xlim(6, 11)
# ax.semilogy()
ax.legend()
ax.set_xlabel(r"${\rm log_{10}} \; \mathrm{age} \,/\, \mathrm{Gyr}$")
ax.set_ylabel(r"SFH")
# plt.show()
plt.savefig(f'plots/sfh_tau.png', bbox_inches='tight', dpi=200)


def get_photo(z, fc, log10m, grid, tau=20 * Myr):
    gal = create_galaxy(z, 10**log10m[0], grid, tau)
    sed = gal.stars.get_spectra(model)
    sed.get_fnu(cosmo, z, igm=Madau96)
    return sed.get_photo_fnu(fc)


# with Pool(processes=nprocs) as pool:
#     args = [(z, fc, log10m, grid, sfh, metal_dist) 
#     for j, z in enumerate(redshifts)]
#     output = pool.starmap(get_photo, args)

output = [get_photo(z, fc, log10m, grid) for z in redshifts]

mass_ratios = 10**log10m / 10**log10m[0]

# Save output
for tau in [-20, -50, -100, -500, -1000, -2000, -4000]:

    output = [get_photo(z, fc, log10m, grid, tau=tau * Myr) for z in redshifts]

    for code in fc.filter_codes: 
        flux = np.zeros((len(log10m), len(redshifts)))
        for i, mass_ratio in enumerate(mass_ratios):
            for j, z in enumerate(redshifts):
                flux[i,j] = output[j][code].value * mass_ratio
    
        np.savetxt(f'data/flux_grid_{code.split('/')[1]}_tau_m{-1 * tau}.txt', flux)

# Plot
fig, ax = plt.subplots(1, 1, figsize=(6, 5))

for m, _flux in zip(log10m, flux[::20,:]):
    ax.plot(redshifts, _flux)

ax.set_ylim(-1, 4)
ax.set_xlabel("Redshift")
ax.set_ylabel("Flux [nJy]")
# ax.legend()
plt.show()
# plt.savefig("plots/flux_vs_redshift_fixed_mass_F277W.png", dpi=200)
# plt.close()
