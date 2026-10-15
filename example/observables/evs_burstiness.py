"""Impact of galaxy-to-galaxy SFH burstiness on the EVS flux of the brightest galaxy (appendix).

Each base SFH is multiplied by 10**delta(t), delta a damped random walk (rms sig dex, correlation
time tau), and normalised by the ensemble-mean mass, so bursts change both the stellar mass and the
light at fixed halo mass. Delta = log10(F / F_smooth) is applied to the halo counts before the
extreme value calculation, then the stellar fraction as in the main analysis. No galaxy may form
more than f_b M_h: for each f* the mass of a realisation is capped there, and for the f* = 1 limit
every realisation forms exactly f_b M_h, so bursts change only its light-to-mass ratio.
"""
from functools import lru_cache

import numpy as np
import h5py
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.cosmology import Planck15 as cosmo
from unyt import Gyr
from synthesizer.parametric import SFH

from evstats.stats import compute_conf_ints
from evstats.stellar import log10_fs_pdf

GRID = "bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-c23.01-sps"
K = 20000                 # realisations per ensemble
SIG, TAU = 0.24, 50e6     # fiducial burstiness (Pallottini & Ferrara 2023)
BAND = "F356W"
F_B = 0.16
FLUX_CUT = 1.0            # nJy
rng = np.random.default_rng(1)

tab = np.load(f"data/ssp_flux_tables_{GRID}.npz")
T, bands = tab["T"], [b.split(".")[-1] for b in tab["bands"]]
ages = 10 ** tab["log10ages"]
edges = np.concatenate([[0.0], 0.5 * (ages[1:] + ages[:-1])])   # Synthesizer's SFH binning

with h5py.File("../data/evs_all.h5", "r") as hf:
    log10m, F, N, z = (hf[k][:] for k in ("log10m", "F", "N", "z"))
dx = log10m[1] - log10m[0]
whole_sky = (41252.96 * u.deg**2).to(u.arcmin**2)
sky_frac = lambda area: float(area / whole_sky)
COSMOS = sky_frac(0.28 * u.deg**2)

FORMS = {
    "Exponential": ("steelblue", "Exponential_tau-0.1",
                    lambda a: SFH.Exponential(tau=-0.1 * Gyr, max_age=a)),
    "LogNormal": ("darkorange", "LogNormal_tau0.4_peak_age0.05",
                  lambda a: SFH.LogNormal(tau=0.4, peak_age=0.05 * Gyr, max_age=a)),
    "Double power law": ("seagreen", "DoublePowerLaw_peak_age0.1_alpha5_beta-5",
                         lambda a: SFH.DoublePowerLaw(peak_age=0.1 * Gyr, alpha=5, beta=-5, max_age=a)),
}


@lru_cache
def flux_grid(tag, band=BAND):
    inst = "MIRI" if band == "F770W" else "NIRCam"
    return np.loadtxt(f"data/flux_grid_{inst}.{band}_{tag}_{GRID}.txt")


def delta(i, make, sig=SIG, tau=TAU):
    """Delta (K, bands) and log10 stellar mass offsets (K,) for bursty realisations of an SFH."""
    tmax = cosmo.age(z[i]).to("yr").value
    sub = np.unique(np.concatenate([edges[edges < tmax], np.arange(0, tmax, 5e5), [tmax]]))
    fine = np.linspace(0, tmax, 400001)
    sfr = make(tmax / 1e9 * Gyr).get_sfr(fine)
    cum = np.concatenate([[0], np.cumsum(0.5 * (sfr[1:] + sfr[:-1]) * np.diff(fine))])
    mu = np.diff(np.interp(sub, fine, cum))                     # smooth mass per sub-interval
    mid = 0.5 * (sub[1:] + sub[:-1])
    j = np.searchsorted(edges, mid) - 1                          # grid age bin of each sub-interval
    a = np.exp(-np.diff(mid, prepend=mid[0]) / tau)
    x = rng.normal(0, sig, K)
    M = np.zeros((len(ages), K))
    for s in range(len(mu)):
        x = a[s] * x + sig * np.sqrt(1 - a[s] ** 2) * rng.normal(size=K)
        M[j[s]] += mu[s] * 10 ** x
    tot = M.sum(0)
    m0 = np.bincount(j, mu, len(ages))
    with np.errstate(divide="ignore", invalid="ignore"):
        D = np.log10((M.T / tot.mean()) @ T[i] / (m0 / m0.sum() @ T[i]))
    return D, np.log10(tot / tot.mean())


def counts(i, D=None):
    """Haloes (all sky) above each log10 M_h, or galaxies above the equivalent flux given offsets D."""
    Ngt = (N[i, 0] - F[i]).astype(float)
    if D is None:
        return Ngt
    h, e = np.histogram(D, np.arange(D.min() - dx, D.max() + dx, dx))
    c = 0.5 * (e[1:] + e[:-1])
    return sum(w * np.interp(log10m - ci, log10m, Ngt, left=Ngt[0], right=0)
               for w, ci in zip(h / h.sum(), c) if w > 0)


v_fs = np.arange(-8, dx / 2, dx)                 # log10 f*, as in apply_fs_distribution
w_fs = log10_fs_pdf(v_fs, "lognormal") * dx
v_fs, w_fs = v_fs[w_fs > 1e-12], w_fs[w_fs > 1e-12]


def halo_pdf(i, D, sky):
    return np.gradient(np.exp(-sky * counts(i, D)), dx)


def mstar_pdf(i, D, lm, sky):
    """Stellar mass PDF over the f* distribution, capping each realisation at f_b M_h (f* 10^lm <= 1)."""
    p0 = halo_pdf(i, D, sky)
    out = np.zeros_like(log10m)
    for v, w in zip(v_fs, w_fs):
        p = p0 if D is None or v + lm.max() <= 0 else halo_pdf(i, D - np.maximum(0, lm + v), sky)
        out += w * np.interp(log10m - v - np.log10(F_B), log10m, p, left=0, right=0)
    return out


def flux_ci(i, D=None, lm=None, tag=FORMS["Exponential"][1], sky=COSMOS, fstar_one=False):
    """EVS flux confidence intervals (nJy); fstar_one gives the f* = 1 limit, at fixed mass f_b M_h."""
    g = flux_grid(tag)[:, i]
    if fstar_one:
        pdf = halo_pdf(i, None if D is None else D - lm, sky)
        return compute_conf_ints(pdf, log10m, np.interp(log10m + np.log10(F_B), log10m, g))
    return compute_conf_ints(mstar_pdf(i, D, lm, sky), log10m, g)


def shift(i, D, lm, **kw):
    """Shift (dex) of the median and +3 sigma EVS flux due to the offsets D."""
    c0, c1 = flux_ci(i, **kw), flux_ci(i, D, lm, **kw)
    return np.where(c0[[3, 6]] > FLUX_CUT, np.log10(c1[[3, 6]] / c0[[3, 6]]), np.nan)


half = lambda x: 0.5 * np.subtract(*np.percentile(x, [84, 16]))
ib = bands.index(BAND)
iz = lambda zt: int(np.argmin(abs(z - zt)))

# Numbers quoted in the appendix: burstiness parameters, exponential SFH, z = 10, COSMOS-Web
print(f"sig  tau  | sigma(log M*)  sigma(Delta_{BAND})  median  +3sig   (z = {z[iz(10)]:.1f})")
for sig in (0.12, 0.24, 0.5):
    for tau in (10e6, 50e6, 100e6):
        D, lm = delta(iz(10), FORMS["Exponential"][2], sig, tau)
        s = shift(iz(10), D[:, ib], lm)
        print(f"{sig:.2f} {tau / 1e6:4.0f} | {half(lm):.2f}  {half(D[:, ib]):.2f}  {s[0]:+.2f}  {s[1]:+.2f}")

# Brightest-galaxy host halo, fiducial burstiness, z = 10, COSMOS-Web
i = iz(10)
D10, _ = delta(i, FORMS["Exponential"][2])
Ngt = counts(i)
h, e = np.histogram(D10[:, ib], np.arange(D10[:, ib].min() - dx, D10[:, ib].max() + dx, dx))
c, w = 0.5 * (e[1:] + e[:-1]), h / h.sum()
NX10 = counts(i, D10[:, ib])
NX = lambda m: np.interp(m, log10m, NX10)
p_host = -np.gradient(Ngt, dx) * sum(wk * np.exp(-COSMOS * NX(log10m + ck)) for wk, ck in zip(w, c) if wk > 0)
cdf = np.cumsum(p_host) / p_host.sum()
mmax = np.interp(0.5, np.exp(-COSMOS * Ngt), log10m)
rank = 1 + COSMOS * Ngt
print(f"host: median offset {np.interp(0.5, cdf, log10m) - mmax:+.2f} dex (2.5th pct "
      f"{np.interp(0.025, cdf, log10m) - mmax:+.2f}); rank median {np.interp(0.5, cdf, rank):.1f}, "
      f"97.5th pct {np.interp(0.025, cdf, rank):.0f}")

# Full sky, and the f* = 1 limit for the Castellano et al. (2025) area at z = 16.3
for zt in (10, 14):
    D, lm = delta(iz(zt), FORMS["Exponential"][2])
    print(f"full sky z={zt}: shift [median, +3sig] =", np.round(shift(iz(zt), D[:, ib], lm, sky=1.0), 2))
i = iz(16.3)
D16, lm16 = delta(i, FORMS["Exponential"][2])
D16 = D16[:, ib]
sky = sky_frac(615 * u.arcmin**2)
f1 = lambda D=None, lm=None: flux_ci(i, D, lm, sky=sky, fstar_one=True)[6]
print(f"z={z[i]:.1f}, 615 arcmin2: +3sig shift {shift(i, D16, lm16, sky=sky)[1]:+.2f}, "
      f"f*=1 +3sig shift {np.log10(f1(D16, lm16) / f1()):+.2f}, "
      f"realisations above the ensemble-mean mass {np.mean(lm16 > 0):.2f}")

# Shift curves vs redshift, F356W, COSMOS-Web (slow, so cached)
CACHE = "data/evs_burstiness_shifts_capped.npz"
zsel = np.where((z >= 2) & (z <= 18))[0]


def curve(make, tag=FORMS["Exponential"][1], **kw):
    out = []
    for i in zsel:
        D, lm = delta(i, make, **kw)
        out.append(shift(i, D[:, ib], lm, tag=tag))
    return np.array(out)


try:
    curves = dict(np.load(CACHE))
except FileNotFoundError:
    curves = {f"{name}_{tau / 1e6:.0f}": curve(make, tag, tau=tau)
              for name, (_, tag, make) in FORMS.items() for tau in (10e6, 50e6, 100e6)}
    curves |= {f"sig{sig}": curve(FORMS["Exponential"][2], sig=sig) for sig in (0.12, 0.5)}
    curves[f"sig{SIG}"] = curves["Exponential_50"]
    np.savez(CACHE, **curves)

fig, axes = plt.subplots(2, 1, figsize=(5, 7), sharex=True, layout="constrained")

ax = axes[0]
for name, (col, _, _) in FORMS.items():
    s = {tau: curves[f"{name}_{tau}"] for tau in (10, 50, 100)}
    ax.fill_between(z[zsel], s[10][:, 0], s[100][:, 0], color=col, alpha=0.25, lw=0)
    ax.plot(z[zsel], s[50][:, 0], color=col, lw=2, label=name)
    ax.plot(z[zsel], s[50][:, 1], color=col, lw=1.3, ls="--")
ax.plot([], [], color="black", lw=1.3, ls="--", label=r"$+3\sigma$")
ax.legend(frameon=False, fontsize=10, title=rf"$\sigma_\delta = {SIG}$ dex, $\tau_\delta = 10$--$100$ Myr",
          title_fontsize=10, loc="upper left", ncol=2, columnspacing=1.2)
ax.set_ylim(0, 0.22)

ax = axes[1]
for sig, col in zip((0.12, 0.24, 0.5), plt.get_cmap("plasma")((0.15, 0.5, 0.8))):
    ax.plot(z[zsel], curves[f"sig{sig}"][:, 0], color=col, lw=2, label=rf"$\sigma_\delta = {sig}$ dex")
    ax.plot(z[zsel], curves[f"sig{sig}"][:, 1], color=col, lw=1.3, ls="--")
ax.legend(frameon=False, fontsize=10, title=r"Exponential, $\tau_\delta = 50$ Myr", title_fontsize=10,
          loc="upper left")
ax.set_ylim(0, 0.6)

for ax in axes:
    ax.set_ylabel(rf"$\Delta \log_{{10}} F_\nu^{{\rm {BAND}}}\ \mathrm{{[dex]}}$", size=12)
axes[1].set_xlim(2, 18)
axes[1].set_xlabel("$z$", size=12)

plt.savefig("plots/evs_burstiness.pdf", bbox_inches="tight")
print("wrote plots/evs_burstiness.pdf")
