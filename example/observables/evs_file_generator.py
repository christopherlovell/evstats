"""Generate flux grids over SFH parameterisations, SPS models and metallicities.

One flux grid per (SFH, metallicity, SPS model, JWST band), written to
data/flux_grid_{band}_{sfh_tag}{z_tag}_{grid_name}.txt. Uses the locally
installed grids (bpass here is 0.1,300.0, fsps is mist-miles); incident
emission is stellar only, intrinsic includes photoionisation.

Usage: evs_file_generator.py [SFH_SUBSTRING] [GRID_SUBSTRING] [METALLICITY]

All arguments are optional and filter the tables below, e.g.
`evs_file_generator.py Exponential_tau-0.1 cloudy 0.01` for the fiducial
exponential SFH, BPASS + nebular emission and the fiducial metallicity. With
no arguments every combination is generated, which is slow.
"""
import sys

import numpy as np
import h5py
from unyt import Msun, angstrom, Gyr
from astropy.cosmology import Planck15 as cosmo

from synthesizer.grid import Grid
from synthesizer.parametric import SFH, Stars, ZDist
from synthesizer import galaxy
from synthesizer.instruments import FilterCollection
from synthesizer.emission_models import IncidentEmission, IntrinsicEmission
from synthesizer.emission_models.attenuation import Madau96

GRID_DIR = "/home/chris/code/synthesizer_grids/grids"

# (grid, emission model): incident = stellar only; intrinsic = with photoionisation.
GRIDS = [
    ("bc03-2016-Miles_chabrier-0.1,100", lambda g: IncidentEmission(grid=g)),
    ("bc03-2016-Miles_salpeter-0.1,100", lambda g: IncidentEmission(grid=g)),
    ("bc03-2016-Miles_kroupa-0.1,100", lambda g: IncidentEmission(grid=g)),
    ("fsps-3.2-mist-miles_chabrier03-0.5,120", lambda g: IncidentEmission(grid=g)),
    ("bpass-2.2.1-bin_chabrier03-0.1,300.0", lambda g: IncidentEmission(grid=g)),
    ("bc03-2016-Miles_chabrier-0.1,100_cloudy-c23.01-sps",
     lambda g: IntrinsicEmission(grid=g, fesc=0.0)),
    ("fsps-3.2-mistmiles_chabrier03-0.5,120_cloudy-c23.01-sps",
     lambda g: IntrinsicEmission(grid=g, fesc=0.0)),
    ("bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-c23.01-sps",
     lambda g: IntrinsicEmission(grid=g, fesc=0.0)),
]

# Tags are written out in full so they match the flux grid filenames directly.
SFHS = [
    ("Exponential_tau-0.03", lambda a: SFH.Exponential(tau=-0.03 * Gyr, max_age=a)),
    ("Exponential_tau-0.05", lambda a: SFH.Exponential(tau=-0.05 * Gyr, max_age=a)),
    ("Exponential_tau-0.1", lambda a: SFH.Exponential(tau=-0.1 * Gyr, max_age=a)),
    ("Exponential_tau-0.3", lambda a: SFH.Exponential(tau=-0.3 * Gyr, max_age=a)),
    ("Exponential_tau-1.0", lambda a: SFH.Exponential(tau=-1.0 * Gyr, max_age=a)),
    ("LogNormal_tau0.25_peak_age0.03",
     lambda a: SFH.LogNormal(tau=0.25, peak_age=0.03 * Gyr, max_age=a)),
    ("LogNormal_tau0.4_peak_age0.05",
     lambda a: SFH.LogNormal(tau=0.4, peak_age=0.05 * Gyr, max_age=a)),
    ("LogNormal_tau0.7_peak_age0.08",
     lambda a: SFH.LogNormal(tau=0.7, peak_age=0.08 * Gyr, max_age=a)),
    ("DoublePowerLaw_peak_age0.05_alpha10_beta-10",
     lambda a: SFH.DoublePowerLaw(peak_age=0.05 * Gyr, alpha=10, beta=-10, max_age=a)),
    ("DoublePowerLaw_peak_age0.1_alpha5_beta-5",
     lambda a: SFH.DoublePowerLaw(peak_age=0.1 * Gyr, alpha=5, beta=-5, max_age=a)),
    ("DoublePowerLaw_peak_age0.2_alpha1_beta-1",
     lambda a: SFH.DoublePowerLaw(peak_age=0.2 * Gyr, alpha=1, beta=-1, max_age=a)),
]

# Metallicities spanning ~0.007-3 Zsun (4e-2 is the highest in the BPASS grid).
# The fiducial, Z = 0.01, carries an empty tag so its filenames are unadorned.
METALLICITIES = [1e-4, 1e-3, 4e-3, 1e-2, 2e-2, 4e-2]

BANDS = ["JWST/NIRCam.F115W", "JWST/NIRCam.F150W", "JWST/NIRCam.F200W",
         "JWST/NIRCam.F277W", "JWST/NIRCam.F356W", "JWST/NIRCam.F444W",
         "JWST/MIRI.F770W"]

sfh_filter = sys.argv[1] if len(sys.argv) > 1 else ""
grid_filter = sys.argv[2] if len(sys.argv) > 2 else ""
z_filter = sys.argv[3] if len(sys.argv) > 3 else ""

with h5py.File("../data/evs_all.h5", "r") as hf:
    log10m = hf["log10m"][:]
    z = hf["z"][:]


def create_galaxy(zval, m, grid, make_sfh, zdist):
    sfh = make_sfh(cosmo.age(zval) * Gyr)
    stars = Stars(grid.log10ages, grid.metallicities, sf_hist=sfh,
                  metal_dist=zdist, initial_mass=m * Msun)
    return galaxy(stars=stars, redshift=zval)


# Grids are loaded once and reused across the SFHs, so they set the outer loop.
for grid_name, make_model in GRIDS:
    if grid_filter not in grid_name:
        continue

    grid = Grid(grid_name, grid_dir=GRID_DIR, new_lam=np.logspace(2.3, 5, 500) * angstrom)
    model = make_model(grid)
    fc = FilterCollection(BANDS, new_lam=grid.lam)

    for sfh_tag, make_sfh in SFHS:
        if sfh_filter not in sfh_tag:
            continue

        for Z in METALLICITIES:
            if z_filter not in str(Z):
                continue
            z_tag = "" if Z == 0.01 else f"_Z{Z}"
            zdist = ZDist.DeltaConstant(metallicity=Z)

            flux = {b: np.zeros((len(log10m), len(z))) for b in BANDS}
            for i, zval in enumerate(z):
                gal = create_galaxy(zval, 10 ** log10m[0], grid, make_sfh, zdist)
                sed = gal.stars.get_spectra(model)
                sed.get_fnu(cosmo, zval, igm=Madau96)
                photo = sed.get_photo_fnu(fc)
                for b in BANDS:
                    flux[b][:, i] = photo[b].value * 10 ** (log10m - log10m[0])

            for b in BANDS:
                np.savetxt(
                    f"data/flux_grid_{b.split('/')[-1]}_{sfh_tag}{z_tag}_"
                    f"{grid_name}.txt", flux[b]
                )
            print("wrote grids for", sfh_tag + z_tag, grid_name, flush=True)
