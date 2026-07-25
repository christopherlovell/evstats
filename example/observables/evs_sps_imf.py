import numpy as np
import h5py
import matplotlib.pyplot as plt

import astropy.units as u
from evstats import evs
from evstats.stats import compute_conf_ints


with h5py.File('../data/evs_all.h5', 'r') as hf:
    log10m = hf['log10m'][:]
    f = hf['f'][:]
    F = hf['F'][:]
    N = hf['N'][:]
    z = hf['z'][:]

whole_sky = (41252.96 * u.deg**2).to(u.arcmin**2)
survey_area = 0.28 * u.deg**2
fsky = float(survey_area / whole_sky)
phi_max = evs._apply_fsky(N, f, F, fsky)
redshift_idx = np.arange(len(z))

sfh_tag = "DoublePowerLaw_peak_age0.2_alpha1_beta-1"
bc03_chab = f"{sfh_tag}_bc03-2016-Miles_chabrier-0.1,100"
salpeter = f"{sfh_tag}_bc03-2016-Miles_salpeter-0.1,100"
kroupa = f"{sfh_tag}_bc03-2016-Miles_kroupa-0.1,100"
fsps = f"{sfh_tag}_fsps-3.2-mist-miles_chabrier03-0.5,120"
bpass = f"{sfh_tag}_bpass-2.2.1-bin_chabrier03-0.1,300.0"
bpass_neb = f"{sfh_tag}_bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-c23.01-sps"

# Each row shows the dex offset from its fiducial (drawn as the zero line).
rows = [
    dict(label="SPS", fid=bpass, fid_name="BPASS",
         models={"BC03": (bc03_chab, "coral"), "FSPS": (fsps, "steelblue"),
                 "BPASS+nebular": (bpass_neb, "purple")}),
    dict(label="IMF", fid=bc03_chab, fid_name="Chabrier",
         models={"Salpeter": (salpeter, "steelblue"), "Kroupa": (kroupa, "mediumseagreen")}),
]
bands = ['NIRCam.F115W', 'NIRCam.F277W', 'NIRCam.F444W']
FLOOR = 1.0  # nJy; mask where the flux drops out (F115W Lyman break)


def median_flux(tag, band):
    """Median EVS flux (nJy) vs redshift for a flux grid."""
    g = np.loadtxt(f"data/flux_grid_{band}_{tag}.txt")
    return np.vstack([compute_conf_ints(phi_max[i], g[:, i]) for i in redshift_idx])[:, 3]


def dex_diff(Fm, Ff):
    """log10(Fm / Ff), masked where either flux is undetectable."""
    d = np.full_like(Ff, np.nan)
    ok = (Fm > FLOOR) & (Ff > FLOOR)
    d[ok] = np.log10(Fm[ok]) - np.log10(Ff[ok])
    return d


fig, axes = plt.subplots(2, 3, figsize=(13, 7.6), sharex=True, sharey=True, layout="constrained")

for i, row in enumerate(rows):
    for j, band in enumerate(bands):
        ax = axes[i, j]
        Ffid = median_flux(row["fid"], band)
        ax.axhline(0, color="grey", ls="--", lw=1, zorder=1)
        for tag, color in row["models"].values():
            ax.plot(z, dex_diff(median_flux(tag, band), Ffid), color=color, lw=2.2, zorder=3)
        ax.set_xlim(2, 18)
        ax.set_ylim(-0.3, 0.1)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if i == 0:
            ax.text(0.05, 0.95, band.split('.')[-1], size=12, color='black',
                    va='top', transform=ax.transAxes)

    handles = [plt.Line2D([0], [0], color="grey", ls="--", lw=1)]
    handles += [plt.Line2D([0], [0], color=c, lw=2.2) for _, c in row["models"].values()]
    labels = [f"{row['fid_name']} (fiducial)"] + list(row["models"].keys())
    axes[i, 0].legend(handles=handles, labels=labels, title=f"{row['label']} models",
                      frameon=False, loc='center right', fontsize=11, title_fontsize=11)

for ax in axes[1, :]:
    ax.set_xlabel('$z$', size=12)
for ax in axes[:, 0]:
    ax.set_ylabel(r"$\Delta \log_{10} F_\nu\ \mathrm{[dex]}$", size=12)

plt.savefig('plots/evs_sps_imf_dpl.png', bbox_inches='tight', dpi=200)
print("wrote plots/evs_sps_imf_dpl.png")
