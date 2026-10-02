import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

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

whole_sky = (41252.96 * u.deg**2).to(u.arcmin**2)
survey_area = 0.28 * u.deg**2
fsky = float(survey_area / whole_sky)
phi_max = evs._apply_fsky(N, f, F, fsky)
redshift_idx = np.arange(len(z))

# Stellar masses from the baryon fraction and a lognormal stellar fraction.
mstar_pdf = np.vstack([apply_fs_distribution(p, log10m, f_b=0.16) for p in phi_max])

sfh_tag = "Exponential_tau-0.1"
bc03_chab = f"{sfh_tag}_bc03-2016-Miles_chabrier-0.1,100"
salpeter = f"{sfh_tag}_bc03-2016-Miles_salpeter-0.1,100"
kroupa = f"{sfh_tag}_bc03-2016-Miles_kroupa-0.1,100"
fsps = f"{sfh_tag}_fsps-3.2-mist-miles_chabrier03-0.5,120"
bpass = f"{sfh_tag}_bpass-2.2.1-bin_chabrier03-0.1,300.0"
bc03_neb = f"{bc03_chab}_cloudy-c23.01-sps"
fsps_neb = f"{sfh_tag}_fsps-3.2-mistmiles_chabrier03-0.5,120_cloudy-c23.01-sps"
bpass_neb = f"{bpass}_cloudy-c23.01-sps"

# The metallicity row varies the (delta function) stellar metallicity about the
# fiducial Z = 0.01, holding the SFH and the BPASS+nebular grid fixed.
Z_LABELS = {1e-4: r'$10^{-4}$', 1e-3: r'$10^{-3}$', 4e-3: r'$4\times10^{-3}$',
            2e-2: r'$2\times10^{-2}$', 4e-2: r'$4\times10^{-2}$'}
znorm = LogNorm(vmin=1e-4, vmax=4e-2)
zcmap = plt.get_cmap('viridis')

# Each row shows the dex offset from its fiducial (drawn as the zero line), for
# (nebular, stellar only, colour). BPASS+nebular is the fiducial, so only its
# stellar only counterpart is drawn; the Salpeter and Kroupa grids have no cloudy
# processed counterpart, so the IMF row is stellar only throughout.
rows = [
    dict(title="SPS models", fid=bpass_neb, fid_name="BPASS+nebular",
         models={"BPASS": (None, bpass, "purple"),
                 "BC03": (bc03_neb, bc03_chab, "coral"),
                 "FSPS": (fsps_neb, fsps, "steelblue")}),
    dict(title="IMF", fid=bc03_chab, fid_name="Chabrier",
         models={"Salpeter": (None, salpeter, "steelblue"),
                 "Kroupa": (None, kroupa, "mediumseagreen")}),
    dict(title="Metallicity", fid=bpass_neb, fid_name=r"$Z = 10^{-2}$",
         models={lab: (f"{sfh_tag}_Z{Z}_bpass-2.2.1-bin_chabrier03-0.1,300.0"
                       "_cloudy-c23.01-sps", None, zcmap(znorm(Z)))
                 for Z, lab in Z_LABELS.items()}),
]
bands = ['NIRCam.F115W', 'NIRCam.F277W', 'NIRCam.F444W']

# Once a band drops out (the F115W Lyman break) the flux falls by tens of dex,
# and the difference between two vanishing fluxes is meaningless; truncate there.
FLUX_CUT = 1.0  # nJy


def log10_median(tag, band):
    """log10 of the median EVS flux (nJy) vs redshift for a flux grid."""
    g = np.loadtxt(f"data/flux_grid_{band}_{tag}.txt")
    ci = np.vstack([compute_conf_ints(mstar_pdf[i], log10m, g[:, i]) for i in redshift_idx])[:, 3]
    return np.where(ci > FLUX_CUT, np.log10(np.maximum(ci, FLUX_CUT)), np.nan)


fig, axes = plt.subplots(3, 3, figsize=(13, 11.4), sharex=True, sharey=True, layout="constrained")

for i, row in enumerate(rows):
    for j, band in enumerate(bands):
        ax = axes[i, j]
        Ffid = log10_median(row["fid"], band)
        ax.axhline(0, color="grey", ls=":", lw=1, zorder=1)
        for neb, stellar, color in row["models"].values():
            for tag, ls in ((neb, "-"), (stellar, "--")):
                if tag is None:
                    continue
                ax.plot(z, log10_median(tag, band) - Ffid,
                        color=color, ls=ls, lw=2.2, zorder=3)
        ax.set_xlim(2, 18)
        ax.set_ylim(-0.45, 0.3)
        if i == 0:
            ax.text(0.05, 0.95, band.split('.')[-1], size=12, color='black',
                    va='top', transform=ax.transAxes)

    # Swatches take the linestyle actually drawn for that model.
    handles = [plt.Line2D([0], [0], color="grey", ls=":", lw=1)]
    handles += [plt.Line2D([0], [0], color=c, ls="-" if neb else "--", lw=2.2)
                for neb, _, c in row["models"].values()]
    labels = [f"{row['fid_name']} (fiducial)"] + list(row["models"].keys())
    if any(neb and stellar for neb, stellar, _ in row["models"].values()):
        handles += [plt.Line2D([0], [0], color="black", lw=2.2),
                    plt.Line2D([0], [0], color="black", ls="--", lw=2.2)]
        labels += ["with nebular", "stellar only"]
    axes[i, 0].legend(handles=handles, labels=labels, title=row["title"],
                      frameon=False, loc='lower right', markerfirst=False,
                      alignment='right', fontsize=11, title_fontsize=11)

for ax in axes[-1, :]:
    ax.set_xlabel('$z$', size=12)
for ax in axes[:, 0]:
    ax.set_ylabel(r"$\Delta \log_{10} F_\nu\ \mathrm{[dex]}$", size=12)

plt.savefig('plots/evs_sps_imf_z_exp.png', bbox_inches='tight', dpi=200)
print("wrote plots/evs_sps_imf_z_exp.png")
