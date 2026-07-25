"""Generate SPS-varied flux grids for the fiducial DPL SFH (Figs 2 & 4).

One flux grid per (SPS model, JWST band), incident emission (no photoionisation),
tagged {SFH_TAG}_{grid_name} to match evs_SPS.py / hypersurface_confs.py.
Uses the locally installed grids (bpass here is 0.1,300.0, fsps is mist-miles).
"""
import numpy as np
import h5py
from unyt import Msun, angstrom, Gyr
from astropy.cosmology import Planck18 as cosmo

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
    ("bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-c23.01-sps",
     lambda g: IntrinsicEmission(grid=g, fesc=0.0)),
]
BANDS = ["JWST/NIRCam.F115W", "JWST/NIRCam.F150W", "JWST/NIRCam.F277W", "JWST/NIRCam.F444W"]
SFH_TAG = "DoublePowerLaw_peak_age0.2_alpha1_beta-1"
SFH_PARAMS = dict(peak_age=0.2 * Gyr, alpha=1, beta=-1)

with h5py.File("../data/evs_all.h5", "r") as hf:
    log10m = hf["log10m"][:]
    z = hf["z"][:]


def create_galaxy(zval, m, grid):
    sfh = SFH.DoublePowerLaw(max_age=cosmo.age(zval) * Gyr, **SFH_PARAMS)
    stars = Stars(grid.log10ages, grid.metallicities, sf_hist=sfh,
                  metal_dist=ZDist.Normal(mean=0.01, sigma=0.005),
                  initial_mass=m * Msun)
    return galaxy(stars=stars, redshift=zval)


for grid_name, make_model in GRIDS:
    grid = Grid(grid_name, grid_dir=GRID_DIR, new_lam=np.logspace(2.3, 5, 500) * angstrom)
    model = make_model(grid)
    fc = FilterCollection(BANDS, new_lam=grid.lam)

    flux = {b: np.zeros((len(log10m), len(z))) for b in BANDS}
    for i, zval in enumerate(z):
        gal = create_galaxy(zval, 10 ** log10m[0], grid)
        sed = gal.stars.get_spectra(model)
        sed.get_fnu(cosmo, zval, igm=Madau96)
        photo = sed.get_photo_fnu(fc)
        for b in BANDS:
            flux[b][:, i] = photo[b].value * 10 ** (log10m - log10m[0])

    for b in BANDS:
        np.savetxt(f"data/flux_grid_{b.split('/')[-1]}_{SFH_TAG}_{grid_name}.txt", flux[b])
    print("wrote grids for", grid_name, flush=True)
