import numpy as np
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
import h5py
from unyt import Hz, Msun, Myr, erg, nJy, s, angstrom, Gyr
import matplotlib as plt

from synthesizer.grid import Grid
from synthesizer.parametric import SFH, Stars, ZDist
from synthesizer import galaxy
from synthesizer.instruments import FilterCollection
from synthesizer.emission_models import IncidentEmission
from synthesizer.emission_models.attenuation import Madau96


with h5py.File('../data/evs_all.h5','r') as hf:
    log10m = hf['log10m'][:]
    f = hf['f'][:]
    F = hf['F'][:]
    N = hf['N'][:]
    z = hf['z'][:]

# nprocs = 2

# Loading grid
grid_name = "test_grid"
grid_dir = "../../../synthesizer/tests/test_grid/"
grid = Grid(
    grid_name,
    grid_dir=grid_dir,
    new_lam=np.logspace(2.3, 5, 500) * angstrom
)

model = IncidentEmission(grid=grid)

#Filter Code List
filter_codes = [
    f"JWST/{filt}" for filt in
    ['NIRCam.F277W', 'NIRCam.F115W', 'NIRCam.F444W', 'MIRI.F770W']
]
fc = FilterCollection(filter_codes, new_lam=grid.lam)


#Creating galaxy, checks which sfh_model called and params
def create_galaxy(z, m, grid, sfh_type, sfh_params):
    max_age = cosmo.age(z) * Gyr 
    
    if sfh_type == "Exponential":
        sfh = SFH.Exponential(
            tau = sfh_params["tau"],
            max_age = max_age
        )
    elif sfh_type == "LogNormal":
        sfh = SFH.LogNormal(
            tau = sfh_params["tau"],
            peak_age = sfh_params["peak_age"],
            max_age = max_age
        )
    elif sfh_type == "DoublePowerLaw":
        sfh = SFH.DoublePowerLaw(
            peak_age = sfh_params["peak_age"],
            alpha = sfh_params["alpha"],
            beta = sfh_params["beta"],
            max_age = max_age
        )

    metal_dist = ZDist.Normal(mean=0.01, sigma=0.005)
    stars = Stars(
        grid.log10age,
        grid.metallicity,
        sf_hist=sfh,
        metal_dist=metal_dist,
        initial_mass=m * Msun
    )

    return galaxy(stars=stars, redshift=z)

# Hardcoded Values to loop through, MAYBE add constant, other types of exp. etc.
# Should be able to comment out a model to remove, or add into params into here and if statement above
sfh_models = {
    "Exponential": [{"tau": tau * Gyr} for tau in [-0.02, -0.05, -0.1, -0.5, -1.0]],
    "LogNormal": [{"tau": 0.3, "peak_age": 0.03 * Gyr}, {"tau": 0.5, "peak_age": 0.05 * Gyr}, {"tau": 0.8, "peak_age": 0.08 * Gyr}],
    "DoublePowerLaw": [{"peak_age": 0.05 * Gyr, "alpha": 10, "beta": -10}, {"peak_age": 0.1 * Gyr, "alpha": 5, "beta": -5}, {"peak_age": 0.2 * Gyr, "alpha": 1, "beta": -1}]
}



# Loops over SFH types and parameters + plots
# Unsure on some plotting/colour code - used AI to bugfix but end result *looks* nice?

for sfh_type, param_list in sfh_models.items():
    for params in param_list:
        
        # Create a tag for this model
        tag = sfh_type
        for k, v in params.items():
            if hasattr(v, 'to_value'):
                val = np.round(v.to_value(Gyr), 3)
            else:
                val = v
            tag += f"_{k}{val}"
        
        print(f"\nProcessing: {tag}")
        
        # Loop over each band
        for band in fc.filter_codes:
            
            # Initialize flux grid for THIS band
            flux_grid = np.zeros((len(log10m), len(z)))
            
            # Loop over redshifts
            for i, zval in enumerate(z):
                # Create galaxy at this redshift with lowest mass
                gal = create_galaxy(zval, 10**log10m[0], grid, sfh_type, params)
                
                # Get spectrum and photometry
                sed = gal.stars.get_spectra(model)
                sed.get_fnu(cosmo, zval, igm=Madau96)
                fnu = sed.get_photo_fnu(fc)[band].value
                
                # Scale to all masses
                flux_grid[:, i] = fnu * 10**(log10m - log10m[0])
            
            # Save flux grid for this band AFTER finishing all redshifts
            band_name = band.split('/')[-1]
            filename = f"data/flux_grid_{band_name}_{tag}.txt"
            np.savetxt(filename, flux_grid)
            print(f"  Saved: {filename}")
