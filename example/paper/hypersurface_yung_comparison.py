"""Halo-mass EVS: Behroozi+13 (Planck15) vs Yung+24 (GUREFT).

Fiducial is Behroozi+13 with the old linear transfer extrapolation. Top: its
median (dotted) plus EVS bands/medians for Behroozi (EH extrapolation) and Yung.
Bottom: median fractional difference of those two relative to the fiducial.
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
import astropy.units as u

from evstats import evs
from evstats.stats import compute_conf_ints

whole_sky = (41252.96 * u.deg**2).to(u.arcmin**2)
survey_area = 38 * u.arcmin**2
fsky = float(survey_area / whole_sky)


def halo_cis(fname):
    """Per-redshift most-massive-halo mass CIs (and z grid) for one EVS grid."""
    with h5py.File(fname, 'r') as hf:
        log10m, f, F, N, z = (hf[k][:] for k in ('log10m', 'f', 'F', 'N', 'z'))
    return z, compute_conf_ints(np.asarray(evs._apply_fsky(N, f, F, fsky), float), log10m)


models = {
    'behroozi': dict(file='../data/evs_all.h5', label='Behroozi+13 (EH extrap.)',
                     band='mistyrose', mid='lightcoral', line='brown'),
    'yung':     dict(file='../data/evs_lightcone_yung24.h5', label='Yung+24 (GUREFT)',
                     band='powderblue', mid='steelblue', line='navy'),
}
for m in models.values():
    m['z'], m['CI'] = halo_cis(m['file'])

# Fiducial: Behroozi with the old linear transfer extrapolation.
zref, CIref = halo_cis('../data/evs_all_ehFalse.h5')

zb, CIb = models['behroozi']['z'], models['behroozi']['CI']
zy, CIy = models['yung']['z'], models['yung']['CI']
d_behroozi = 10 ** (CIb[:, 3] - np.interp(zb, zref, CIref[:, 3])) - 1
d_yung = 10 ** (CIy[:, 3] - np.interp(zy, zref, CIref[:, 3])) - 1

fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(5, 6.5), sharex=True,
                               gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05})

for m in models.values():
    z, CI = m['z'], m['CI']
    ax0.fill_between(z, CI[:, 0], CI[:, 6], color=m['band'], alpha=0.55)
    ax0.fill_between(z, CI[:, 2], CI[:, 4], color=m['mid'], alpha=0.55)
    ax0.plot(z, CI[:, 3], color=m['line'], lw=1.8, label=m['label'])
ax0.plot(zref, CIref[:, 3], color='red', lw=1.8, ls='dotted',
         label='Behroozi+13 (linear extrap.)')

ax0.set_ylim(8, 13)
ax0.set_ylabel(r'$\mathrm{log_{10}}(M_{\mathrm{max}} \,/\, M_{\odot})$', size=15)
ax0.text(0.02, 0.05, r'$A = 38 \; \mathrm{arcmin}^2$', size=12, alpha=0.8,
         transform=ax0.transAxes)
ax0.legend(frameon=False, loc='upper right', fontsize=11)

# --- bottom: median fractional difference vs the linear-extrapolation fiducial ---
ax1.axhline(0, color='red', lw=1, ls='dotted')
ax1.plot(zy, d_yung, color='navy', lw=1.8, label='Yung+24')
ax1.plot(zb, d_behroozi, color='brown', lw=1.8, label='Behroozi+13 (EH extrap.)')
ax1.set_xlim(6, 18)
ax1.set_xlabel(r'$z$', size=17)
ax1.set_ylabel(r'$\Delta M_{\mathrm{max}} / M_{\mathrm{Beh}}^{\mathrm{lin}}$', size=11)
ax1.legend(frameon=False, ncol=2, fontsize=10)

plt.savefig('plots/evs_hmf_comparison.pdf', bbox_inches='tight', dpi=200)
plt.close()
print('wrote plots/evs_hmf_comparison.pdf')
