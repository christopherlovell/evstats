import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import astropy.units as u
from evstats import evs
from evstats.stats import compute_conf_ints
from evstats.stellar import apply_fs_distribution


with h5py.File('../data/evs_all.h5', 'r') as hf:
    log10m = hf['log10m'][:]
    f = hf['f'][:]
    F = hf['F'][:]
    N = hf['N'][:]
    z = hf['z'][:]

_obs_str = 'SFH_exp_shade'

whole_sky = (41252.96 * u.deg**2).to(u.arcmin**2)
survey_area = 0.28 * u.degree**2
fsky = float(survey_area / whole_sky)
phi_max = evs.apply_fsky(N, f, F, fsky)
redshift_idx = np.arange(len(z))

# Stellar masses from the baryon fraction and a lognormal stellar fraction.
mstar_pdf = np.vstack([apply_fs_distribution(p, log10m, f_b=0.16) for p in phi_max])

# Fiducial SPS model, common to every SFH shown here.
grid_tag = "bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-c23.01-sps"

# Each parametric form: fiducial grid (solid line) + parameter sweep (shaded range).
forms = {
    "Exponential": dict(
        color="steelblue", fid="Exponential_tau-0.1",
        sweep=["Exponential_tau-0.03", "Exponential_tau-0.05", "Exponential_tau-0.1",
               "Exponential_tau-0.3", "Exponential_tau-1.0"]),
    "LogNormal": dict(
        color="darkorange", fid="LogNormal_tau0.4_peak_age0.05",
        sweep=["LogNormal_tau0.25_peak_age0.03", "LogNormal_tau0.4_peak_age0.05",
               "LogNormal_tau0.7_peak_age0.08"]),
    "Double power law": dict(
        color="seagreen", fid="DoublePowerLaw_peak_age0.1_alpha5_beta-5",
        sweep=["DoublePowerLaw_peak_age0.2_alpha1_beta-1",
               "DoublePowerLaw_peak_age0.1_alpha5_beta-5",
               "DoublePowerLaw_peak_age0.05_alpha10_beta-10"]),
}


def median_ci(tag, band):
    """log10 of the median EVS flux (nJy) vs redshift for a flux-per-mass table."""
    fpm = np.loadtxt(f"data/flux_per_mass_{band}_{tag}_{grid_tag}.txt")
    ci = np.vstack([compute_conf_ints(mstar_pdf[i], log10m, 10**log10m * fpm[i]) for i in redshift_idx])[:, 3]
    return np.log10(np.where(ci > 0, ci, 1e-30))


# Differences are taken relative to the fiducial exponential, the reference model of
# the SPS/IMF/metallicity figure, and truncated at the lower limit of the flux row.
FLUX_CUT = 0.1  # nJy

fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), sharex=True, sharey="row",
                         height_ratios=[1.6, 1], layout="constrained")

for (ax, dax), band in zip(axes.T, ['NIRCam.F115W', 'NIRCam.F277W', 'NIRCam.F444W']):
    ref = median_ci(forms["Exponential"]["fid"], band)
    ref = np.where(ref > np.log10(FLUX_CUT), ref, np.nan)
    for m in forms.values():
        sweep = np.vstack([median_ci(t, band) for t in m["sweep"]])
        fid = median_ci(m["fid"], band)
        ax.fill_between(z, sweep.min(0), sweep.max(0), color=m["color"], alpha=0.2, lw=0, zorder=1)
        ax.plot(z, fid, color=m["color"], lw=2.2, zorder=3)

        dsweep = np.where(sweep > np.log10(FLUX_CUT), sweep, np.nan) - ref
        dax.fill_between(z, dsweep.min(0), dsweep.max(0), color=m["color"], alpha=0.2, lw=0, zorder=1)
        dax.plot(z, np.where(fid > np.log10(FLUX_CUT), fid, np.nan) - ref,
                 color=m["color"], lw=2.2, zorder=3)

    ax.set_ylim(np.log10(FLUX_CUT), 8)
    ax.text(0.15, 0.95, band.split('.')[-1], size=12, color='black',
            va='top', transform=ax.transAxes)
    dax.set_xlim(2, 18)
    dax.set_xlabel('$z$', size=12)

axes[0, 0].set_ylabel(r"$\log_{10}(F_\nu \,/\, \mathrm{nJy})$", size=12)
axes[1, 0].set_ylabel(r"$\Delta \log_{10} F_\nu\ \mathrm{[dex]}$", size=12)

handles = [plt.Line2D([0], [0], color=m["color"], lw=2.2) for m in forms.values()]
handles.append(mpatches.Patch(color="grey", alpha=0.3))
labels = list(forms.keys()) + ["parameter range"]
axes[0, 0].legend(handles=handles, labels=labels, frameon=False, loc='upper right', fontsize=11)

plt.savefig(f'plots/evs_{_obs_str}.png', bbox_inches='tight', dpi=200)
print(f"wrote plots/evs_{_obs_str}.png")
