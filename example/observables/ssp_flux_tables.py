"""Flux per unit mass of a single-age population in each grid age bin, for every redshift shell.

The flux is linear in the SFH, so for any SFH binned onto the grid ages (unit mass),
flux per unit mass = m @ T[iz]. Same grid, emission model, filters and IGM as
evs_file_generator.py; the fiducial SFHs reproduce its flux tables exactly.
"""
import numpy as np
import h5py
from unyt import Msun, angstrom, Gyr
from astropy.cosmology import Planck15 as cosmo

from synthesizer.grid import Grid
from synthesizer.parametric import SFH, Stars, ZDist
from synthesizer import galaxy
from synthesizer.instruments import FilterCollection
from synthesizer.emission_models import IntrinsicEmission
from synthesizer.emission_models.attenuation import Madau96

GRID = "bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-c23.01-sps"
BANDS = ["JWST/NIRCam.F115W", "JWST/NIRCam.F150W", "JWST/NIRCam.F200W",
         "JWST/NIRCam.F277W", "JWST/NIRCam.F356W", "JWST/NIRCam.F444W",
         "JWST/MIRI.F770W"]

with h5py.File("../data/evs_all.h5", "r") as hf:
    z = hf["z"][:]

grid = Grid(GRID, grid_dir="/home/chris/code/synthesizer_grids/grids",
            new_lam=np.logspace(2.3, 5, 500) * angstrom)
model = IntrinsicEmission(grid=grid, fesc=0.0)
fc = FilterCollection(BANDS, new_lam=grid.lam)
zdist = ZDist.DeltaConstant(metallicity=0.01)
nage = len(grid.log10ages)


def photometry(stars, zval):
    sed = galaxy(stars=stars, redshift=zval).stars.get_spectra(model)
    sed.get_fnu(cosmo, zval, igm=Madau96)
    photo = sed.get_photo_fnu(fc)
    return np.array([photo[b].value for b in BANDS])


def stars(zval):
    """Fiducial exponential SFH at zval (also used for the delta ZDist metallicity weights)."""
    sfh = SFH.Exponential(tau=-0.1 * Gyr, max_age=cosmo.age(zval).value * Gyr)
    return Stars(grid.log10ages, grid.metallicities, sf_hist=sfh, metal_dist=zdist, initial_mass=1 * Msun)


mdist = stars(z[0]).metal_dist
mdist = mdist / mdist.sum()

T = np.zeros((len(z), nage, len(BANDS)))
for i, zval in enumerate(z):
    for j in range(nage):
        sfzh = np.zeros((nage, len(mdist)))
        sfzh[j] = mdist
        T[i, j] = photometry(Stars(grid.log10ages, grid.metallicities, sfzh=sfzh, initial_mass=1 * Msun), zval)

# check against the flux tables for the fiducial exponential SFH, in bands above the Lyman break
fid = np.array([stars(zval).sf_hist @ T[i] for i, zval in enumerate(z)])
fg = np.stack([np.loadtxt(f"data/flux_per_mass_{b.split('/')[-1]}_Exponential_tau-0.1_{GRID}.txt")
               for b in BANDS], axis=1)
ok = fg > 1e-3 * fg.max(axis=1, keepdims=True)
print(f"max |log10(table / flux table)| = {np.abs(np.log10(fid[ok] / fg[ok])).max():.1e}")

np.savez(f"data/ssp_flux_tables_{GRID}.npz", T=T, z=z, log10ages=grid.log10ages,
         bands=[b.split("/")[-1] for b in BANDS])
print(f"wrote data/ssp_flux_tables_{GRID}.npz")
