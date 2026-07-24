import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm, ListedColormap
from astropy.cosmology import Planck18 as cosmo
from unyt import Gyr

from synthesizer.parametric import SFH


def sfr_curve(sfh_type, sfh_params, ages_yr, max_age):
    """Analytic SFH evaluated on ages_yr, normalised to unit area (in Myr)."""
    if sfh_type == "Exponential":
        sfh = SFH.Exponential(tau=sfh_params["tau"], max_age=max_age)
    elif sfh_type == "LogNormal":
        sfh = SFH.LogNormal(tau=sfh_params["tau"], peak_age=sfh_params["peak_age"],
                            max_age=max_age)
    elif sfh_type == "DoublePowerLaw":
        sfh = SFH.DoublePowerLaw(peak_age=sfh_params["peak_age"], alpha=sfh_params["alpha"],
                                 beta=sfh_params["beta"], max_age=max_age)

    sfr = np.asarray(sfh.get_sfr(ages_yr), float)
    return sfr / (sfr.sum() * dage_myr)


# Fiducial + a 10-model sweep spanning the comparison limits (paper Sec 3.1).
frac = np.linspace(0, 1, 10)
exp_tau = np.logspace(np.log10(0.03), np.log10(1.0), 10)  # |tau|, rising exponential
sfh_models = {
    "Exponential": dict(
        cmap="Blues", clabel=r"$\tau \,/\, \mathrm{Gyr}$", lognorm=True, fid_val=0.1,
        vals=exp_tau,
        params=[{"tau": -t * Gyr} for t in exp_tau],
        fid={"tau": -0.1 * Gyr}),
    "LogNormal": dict(
        cmap="Oranges", clabel=r"width $\tau$", lognorm=False, fid_val=0.40,
        vals=0.25 + 0.45 * frac,
        params=[{"tau": t, "peak_age": p * Gyr}
                for t, p in zip(0.25 + 0.45 * frac, 0.03 + 0.05 * frac)],
        fid={"tau": 0.40, "peak_age": 0.05 * Gyr}),
    "DoublePowerLaw": dict(
        cmap="Greens", clabel=r"slope $\alpha = |\beta|$", lognorm=False, fid_val=5,
        vals=10 - 9 * frac,
        params=[{"peak_age": p * Gyr, "alpha": a, "beta": -a}
                for a, p in zip(10 - 9 * frac, 0.05 + 0.15 * frac)],
        fid={"peak_age": 0.1 * Gyr, "alpha": 5, "beta": -5}),
}

max_age = cosmo.age(9)
ages_yr = np.linspace(0, max_age.to_value("yr"), 500, endpoint=False)
age_myr = ages_yr / 1e6
dage_myr = age_myr[1] - age_myr[0]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), layout="constrained")
for ax, (sfh_type, m) in zip(axes, sfh_models.items()):
    cmap = ListedColormap(plt.get_cmap(m["cmap"])(np.linspace(0.25, 0.95, 256)))
    norm = (LogNorm if m["lognorm"] else Normalize)(min(m["vals"]), max(m["vals"]))

    for val, params in zip(m["vals"], m["params"]):
        ax.plot(age_myr, sfr_curve(sfh_type, params, ages_yr, max_age),
                color=cmap(norm(val)), lw=1.1, alpha=0.55)
    ax.plot(age_myr, sfr_curve(sfh_type, m["fid"], ages_yr, max_age),
            color="k", lw=2.4, label="fiducial")

    ax.set_xlim(0, 250)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"$\mathrm{age} \,/\, \mathrm{Myr}$", size=12)
    ax.legend(frameon=False, fontsize=11, loc="upper right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax,
                      location="top", pad=0.02, fraction=0.06)
    cb.set_label(f"{sfh_type}    ({m['clabel']})", size=12)
    cb.ax.axvline(m["fid_val"], color="k", lw=1.6)

axes[0].set_ylabel("normalised SFH", size=12)
plt.savefig("plots/sfh_models_comparison.png", dpi=200, bbox_inches="tight")
print("wrote plots/sfh_models_comparison.png")
