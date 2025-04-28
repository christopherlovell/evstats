import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import Planck18 as cosmo
from unyt import Hz, Msun, Myr, erg, nJy, s, angstrom

from synthesizer import galaxy
from synthesizer.conversions import apparent_mag_to_fnu, fnu_to_lnu
from synthesizer.emission_models import PacmanEmission
from synthesizer.emission_models.attenuation import PowerLaw
from synthesizer.instruments import Filter, FilterCollection
from synthesizer.grid import Grid
from synthesizer.parametric import SFH, Stars, ZDist

# Set up a figure to plot on
fig = plt.figure()
ax_lum = fig.add_subplot(111)

# Logscale for both axes
ax_lum.set_xscale("log")
ax_lum.set_yscale("log")

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

# Decide Redshift
redshift = 12

# Set up the SFH
sfh = SFH.Exponential(20 * Myr, 1000 * Myr)

# Set up the metallicity distribution
metal_dist = ZDist.Normal(mean=0.01, sigma=0.005)

# Mass range
masses = np.logspace(8, 12, num=10)

# List to add luminosities to during looping
luminosities = [] 

# Just 1 filter for now out of "F115W", "F150W", "F444W", "F277W"
filter_codes = [f"JWST/NIRCam.{f}" for f in ["F444W"]]
fc = FilterCollection(filter_codes, new_lam=grid.lam)

# Looping through masses
for mass in masses:
    # Get the stellar population
    exp_stars = Stars(
        grid.log10age,
        grid.metallicity,
        sf_hist=sfh,
        metal_dist=metal_dist,
        initial_mass=mass * Msun,
    )

    # Create the galaxy object
    gal = galaxy(stars=exp_stars, redshift=redshift)

    # Generate spectra of galaxy
    gal_spectra = gal.stars.get_spectra(model)

    # Getting the lum for each mass
    lums = gal_spectra.get_photo_lnu(fc)  # This is a PhotometryCollection

    # Dependant on what filter is being used
    luminosity_filt= lums["JWST/NIRCam.F444W"]

    # Adding to list
    luminosities.append(luminosity_filt)

# Need to do this to plot?
luminosities = np.array(luminosities)

logM = np.log10(masses)
logL = np.log10(luminosities)

coeffs = np.polyfit(logM, logL, 1)
alpha = coeffs[0]
logA = coeffs[1]
A = 10**logA

print(coeffs, alpha, logA, A)
    




# Plots and labels
ax_lum.plot(masses, luminosities, marker='o', linestyle='-')
ax_lum.set_xlabel("Mass [M☉]")
ax_lum.set_ylabel("Luminosity [erg/s/Hz]")
ax_lum.set_xlim(1e8, 1e12)  # Mass range
ax_lum.set_ylim(1e15, 1e35)  # Luminosity range
plt.savefig("luminosity_vs_mass_F444W.png", bbox_inches='tight', dpi=200)

