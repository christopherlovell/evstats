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
from synthesizer.emission_models.attenuation import PowerLaw , Madau96

# Define the grid
grid_name = "test_grid"
grid_dir = "../tests/test_grid/"
grid = Grid(grid_name, grid_dir=grid_dir)


# Set up a figure to plot on
fig = plt.figure()
ax_lum = fig.add_subplot(111)

# Logscale for both axes
ax_lum.set_xscale("log")
ax_lum.set_yscale("log")

# Define the emission model
model = PacmanEmission(
    grid,
    fesc=0.5,
    fesc_ly_alpha=0.5,
    tau_v=0.1,
    dust_curve=PowerLaw(slope=-1),
)

# Set up SFH and metallicity
sfh = SFH.Exponential(20 * Myr, 1000 * Myr)
metal_dist = ZDist.Normal(mean=0.01, sigma=0.005)

# Redshifts and fixed masses
redshifts = np.linspace(7, 18, 20)
fixed_masses = [1e9, 1e10, 1e11]

# Filter
filter_code = "JWST/NIRCam.F277W"
fc = FilterCollection([filter_code], new_lam=grid.lam)

# Storage
flux_vs_z = {}

# Loop over fixed masses
for mass in fixed_masses:
    flux_list = []

    for z in redshifts:
        exp_stars = Stars(
            grid.log10age,
            grid.metallicity,
            sf_hist=sfh,
            metal_dist=metal_dist,
            initial_mass=mass * Msun,
        )

        gal = galaxy(stars=exp_stars, redshift=z)

        # Get flux
        sed = gal.stars.get_spectra(model)
        sed.get_fnu(cosmo, z, igm=Madau96)
        flux = sed.get_photo_fnu(fc)[filter_code]

        flux_list.append(flux)
        print(f"Mass = {mass:.1e} Msun - z = {z:.1f} → Flux = {flux:.2e} nJy")

    flux_vs_z[mass] = np.array(flux_list)

# Plotting
for mass, fluxes in flux_vs_z.items():
    plt.plot(redshifts, fluxes, marker='o', label=f"M = {mass:.0e} Msun")

plt.xlabel("Redshift")
plt.ylabel("Flux [nJy]")
plt.legend()
plt.savefig("flux_vs_redshift_fixed_mass_F277W.png", dpi=200)

