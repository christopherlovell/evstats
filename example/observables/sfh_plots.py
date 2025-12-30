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

# # Loading grid
# grid_name = "test_grid"
# grid_dir = "/home/chris/.local/share/Synthesizer/grids/"

# grid = Grid(
#     grid_name,
#     grid_dir=grid_dir,
#     new_lam=np.logspace(2.3, 5, 500) * angstrom
# )

def create_galaxy(m, ages, met, max_age, sfh_type, sfh_params):
    if sfh_type == "Exponential":
        sfh = SFH.Exponential(
            tau = sfh_params["tau"],
            max_age = max_age
        )
    elif sfh_type == "LogNormal":
        sfh = SFH.LogNormal(
            tau = 0.1,  #  sfh_params["tau"],
            peak_age = sfh_params["peak_age"],
            max_age = max_age
        )
    elif sfh_type == "DoublePowerLaw":
        sfh = SFH.DoublePowerLaw(
            peak_age = sfh_params["peak_age"],
            alpha = 5,  # sfh_params["alpha"],
            beta = -5,  #  sfh_params["beta"],
            max_age = max_age
        )

    stars = Stars(
        ages,
        met,
        sf_hist=sfh,
        metal_dist=met[0],
        initial_mass=m * Msun
    )

    return galaxy(stars=stars)  # , redshift=z)

# Hardcoded Values to loop through, MAYBE add constant, other types of exp. etc.
# Should be able to comment out a model to remove, or add into params into here and if statement above
sfh_models = {
    "Exponential": [
        {"tau": tau * Gyr} for tau in [-0.02, -0.1]],
    "LogNormal": [
        {"peak_age": 0.01 * Gyr},  # "tau": 0.1, 
        {"peak_age": 0.05 * Gyr},  # "tau": 0.3, 
        # {"tau": 0.8, "peak_age": 0.08 * Gyr}
    ],
    "DoublePowerLaw": [
        {"peak_age": 0.03 * Gyr},  # , "alpha": 5, "beta": -5},
        {"peak_age": 0.05 * Gyr},  # , "alpha": 5, "beta": -5},
        # {"peak_age": 0.2 * Gyr, "alpha": 1, "beta": -1}
    ]
}

max_age = cosmo.age(9)  # * Gyr 
ages = np.log10(np.linspace(1e-9, max_age.to_value("yr"), 100))
met = np.array([0.01])

# Loops over SFH types and parameters + plots
# Unsure on some plotting/colour code - used AI to bugfix but end result *looks* nice?

fig, ax = plt.subplots(figsize=(5,4))

total_sfh = sum(len(v) for v in sfh_models.values())
cmap = plt.colormaps['viridis']
colors = ['C0', 'C1', 'C2']  # cmap(np.linspace(0, 1, total_sfh))


for i, (sfh_type, param_list) in enumerate(sfh_models.items()):
    for ls, params in zip(["-", "--", ":"], param_list):
        gal = create_galaxy(1, ages, met, max_age, sfh_type, params)
        param_str = ", ".join(f"{k}={v}" for k,v in params.items())
        ax.step(
            10**gal.stars.log10ages / 1e6,
            gal.stars.sf_hist,
            where="mid",
            label=f"{sfh_type}\n({param_str})",
            color=colors[i],
            linestyle=ls
        )

# ax.set_xlim(6, 11)
ax.set_xlim(0, 2.1e2)
ax.set_xlabel(r"$\mathrm{age} \,/\, \mathrm{Myr}$")
ax.set_ylabel("normalised SFH")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("plots/sfh_models_comparison.png", dpi=200, bbox_inches='tight')
plt.show()






