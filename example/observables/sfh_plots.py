import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm, ListedColormap
from matplotlib.ticker import NullLocator
from astropy.cosmology import Planck15 as cosmo
from unyt import Gyr, Msun, angstrom

from synthesizer.grid import Grid
from synthesizer.parametric import SFH, Stars, ZDist
from synthesizer import galaxy
from synthesizer.emission_models import IntrinsicEmission

GRID_DIR = "/home/chris/code/synthesizer_grids/grids"
GRID_NAME = "bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-c23.01-sps"

max_age = cosmo.age(9).to_value("Gyr") * Gyr


def make_sfh(sfh_type, p):
    if sfh_type == "Exponential":
        return SFH.Exponential(tau=p["tau"], max_age=max_age)
    if sfh_type == "LogNormal":
        return SFH.LogNormal(tau=p["tau"], peak_age=p["peak_age"], max_age=max_age)
    if sfh_type == "DoublePowerLaw":
        return SFH.DoublePowerLaw(peak_age=p["peak_age"], alpha=p["alpha"],
                                  beta=p["beta"], max_age=max_age)


def sfr_curve(sfh_type, p):
    """Analytic SFH on ages_yr, normalised to unit area (in Myr)."""
    sfr = np.asarray(make_sfh(sfh_type, p).get_sfr(ages_yr), float)
    return sfr / (sfr.sum() * dage_myr)


# BPASS + cloudy grid; rest-frame intrinsic (nebular + transmitted) spectra.
grid = Grid(GRID_NAME, grid_dir=GRID_DIR, new_lam=np.logspace(2.3, 5, 1000) * angstrom)
model = IntrinsicEmission(grid=grid, fesc=0.0)
lam_um = grid.lam.to("um").value


def sed_curve(sfh_type, p):
    """Rest-frame L_nu per unit formed mass (erg/s/Hz)."""
    stars = Stars(grid.log10ages, grid.metallicities, sf_hist=make_sfh(sfh_type, p),
                  metal_dist=ZDist.Normal(mean=0.01, sigma=0.005), initial_mass=1 * Msun)
    sed = galaxy(stars=stars, redshift=9).stars.get_spectra(model)
    return np.asarray(getattr(sed.lnu, "value", sed.lnu), float)


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

ages_yr = np.linspace(0, max_age.to_value("yr"), 500, endpoint=False)
age_myr = ages_yr / 1e6
dage_myr = age_myr[1] - age_myr[0]

fig, axes = plt.subplots(2, 3, figsize=(13, 7.6), layout="constrained")
for j, (sfh_type, m) in enumerate(sfh_models.items()):
    ax_sfh, ax_sed = axes[0, j], axes[1, j]
    cmap = ListedColormap(plt.get_cmap(m["cmap"])(np.linspace(0.25, 0.95, 256)))
    norm = (LogNorm if m["lognorm"] else Normalize)(min(m["vals"]), max(m["vals"]))

    lnu_fid = sed_curve(sfh_type, m["fid"])
    ref = np.interp(0.15, lam_um, lnu_fid)  # normalise the panel to the fiducial rest-UV

    for val, params in zip(m["vals"], m["params"]):
        c = cmap(norm(val))
        ax_sfh.plot(age_myr, sfr_curve(sfh_type, params), color=c, lw=1.1, alpha=0.55)
        ax_sed.plot(lam_um, sed_curve(sfh_type, params) / ref, color=c, lw=1.1, alpha=0.55)
    ax_sfh.plot(age_myr, sfr_curve(sfh_type, m["fid"]), color="k", lw=2.4, label="fiducial")
    ax_sed.plot(lam_um, lnu_fid / ref, color="k", lw=2.4, label="fiducial")

    ax_sfh.set_xlim(0, 250)
    ax_sfh.set_ylim(bottom=0)
    ax_sfh.set_xlabel(r"$\mathrm{age} \,/\, \mathrm{Myr}$", size=12)
    ax_sfh.legend(frameon=False, fontsize=11, loc="upper right")

    ax_sed.set_xscale("log")
    ax_sed.set_yscale("log")
    ax_sed.set_xlim(0.1, 5.0)
    ax_sed.set_ylim(2e-2, 20)
    ax_sed.set_xticks([0.1, 0.2, 0.5, 1, 2, 5])
    ax_sed.set_xticklabels(["0.1", "0.2", "0.5", "1", "2", "5"])
    ax_sed.xaxis.set_minor_locator(NullLocator())
    ax_sed.set_xlabel(r"rest wavelength $\,/\, \mu\mathrm{m}$", size=12)

    for ax in (ax_sfh, ax_sed):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax_sfh,
                      location="top", pad=0.02, fraction=0.06)
    cb.set_label(f"{sfh_type}    ({m['clabel']})", size=12)
    cb.ax.axvline(m["fid_val"], color="k", lw=1.6)

axes[0, 0].set_ylabel("normalised SFH", size=12)
axes[1, 0].set_ylabel(r"$L_\nu \,/\, L_\nu^{\mathrm{fid}}(1500\,\mathrm{\AA})$", size=12)
plt.savefig("plots/sfh_models_comparison.png", dpi=200, bbox_inches="tight")
print("wrote plots/sfh_models_comparison.png")
