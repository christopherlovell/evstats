import matplotlib.pyplot as plt
import numpy as np
import h5py
from astropy.cosmology import Planck18 as cosmo
from multiprocessing import Pool
from unyt import Hz, Msun, Myr, erg, nJy, s, angstrom, Gyr
import matplotlib.pyplot as plt

from synthesizer import galaxy
from synthesizer.instruments import FilterCollection
from synthesizer.grid import Grid
from synthesizer.parametric import SFH, Stars, ZDist
from synthesizer.emission_models.attenuation import Madau96
from synthesizer.emission_models import IncidentEmission

# nprocs = 2

# Loading grid
grid_name = "test_grid"
grid_dir = "../../../synthesizer/tests/test_grid/"

grid = Grid(
    grid_name,
    grid_dir=grid_dir,
    new_lam=np.logspace(2.3, 5, 500) * angstrom
)



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

fig, ax = plt.subplots(figsize=(6,5))

total_sfh = sum(len(v) for v in sfh_models.values())
cmap = plt.colormaps['viridis']
colors = cmap(np.linspace(0, 1, total_sfh))
color_idx = 0


for sfh_type, param_list in sfh_models.items():
    for params in param_list:
        gal = create_galaxy(1, 10, grid, sfh_type, params)
        param_str = ", ".join(f"{k}={v}" for k,v in params.items())
        ax.step(
            gal.stars.log10ages,
            gal.stars.sf_hist,
            where="mid",
            label=f"{sfh_type} ({param_str})",
            color=colors[color_idx]
        )
        color_idx += 1

ax.set_xlim(6, 11)
ax.set_xlabel(r"${\rm log_{10}} \; \mathrm{age} \,/\, \mathrm{Gyr}$")
ax.set_ylabel("SFH")
ax.legend(fontsize=6)
plt.tight_layout()
plt.savefig("plots/sfh_models_comparison.png", dpi=200)
plt.show()






