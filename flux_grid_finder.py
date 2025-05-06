import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import Planck18 as cosmo
from unyt import Hz, Msun, Myr, erg, nJy, s, angstrom

from synthesizer import galaxy
from synthesizer.conversions import apparent_mag_to_fnu, fnu_to_lnu
from synthesizer.emission_models import PacmanEmission
from synthesizer.emission_models.attenuation import PowerLaw, Madau96
from synthesizer.instruments import FilterCollection
from synthesizer.grid import Grid
from synthesizer.parametric import SFH, Stars, ZDist

# Define the grid
grid_name = "test_grid"
grid_dir = "../tests/test_grid/"
grid = Grid(grid_name, grid_dir=grid_dir)

# Define the emission model
model = PacmanEmission(
    grid,
    fesc=0.5,
    fesc_ly_alpha=0.5,
    tau_v=0.1,
    dust_curve=PowerLaw(slope=-1),
)

# Set up the SFH and metallicity distribution
sfh = SFH.Exponential(20 * Myr, 1000 * Myr)
metal_dist = ZDist.Normal(mean=0.01, sigma=0.005)

# Mass and redshift ranges
masses = np.logspace(4.01, 17, 1299) #1299
redshift = np.arange(0.1, 19.8, 0.2) #99
logM = np.log10(masses)

# Setting up filter
filter_codes = [f"JWST/NIRCam.{f}" for f in ["F277W"]]
fc = FilterCollection(filter_codes, new_lam=grid.lam)

# creating flux grid (supposedly quicker like this)
fluxes_grid = np.empty((len(redshift), len(masses)))

# Loop over redshifts
for i, z in enumerate(redshift):
    print(f"Redshift z = ", z)

# Loop over mass
    for j, mass in enumerate(masses):
        # Creating Star object (exponential)
        exp_stars = Stars(
            grid.log10age,
            grid.metallicity,
            sf_hist=sfh,
            metal_dist=metal_dist,
            initial_mass=mass * Msun,
        )

        # Creating galaxy object and getting flux
        gal = galaxy(stars=exp_stars, redshift=z)
        sed = gal.stars.get_spectra(model)
        sed.get_fnu(cosmo, z, igm=Madau96)
        flux = sed.get_photo_fnu(fc)["JWST/NIRCam.F277W"]

        fluxes_grid[i, j] = flux



# Save results to file (have to manually move)
np.savez("fluxes_grid_data_3.npz",
         fluxes_grid=fluxes_grid,
         masses=masses,
         redshift=redshift)

print(masses)
print(redshift)

# # Plotting redshift logA relation
# plt.figure()
# plt.plot(redshift, logA_list, marker='o')
# plt.xlabel("Redshift")
# plt.ylabel("Log Flux Normalisation (logA)")
# plt.savefig("flux_normalisation_F277W", dpi=200, bbox_inches='tight')



# # Plotting flux and mass
# ax_lum.set_xlabel("Mass [M☉]")
# ax_lum.set_ylabel("Flux [nJy]")
# ax_lum.legend()
# plt.savefig("flux_mass_F444W.png", bbox_inches='tight', dpi=200)
