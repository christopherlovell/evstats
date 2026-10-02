"""Pre-compute a whole-sky (fsky=1) EVS grid for downstream analysis.

Tabulates the EVS ingredients (N, f, F) for the most massive halo in contiguous
redshift shells, for a chosen hmf mass function and cosmology, and writes them
to data/ as HDF5. Downstream analysis scales this whole-sky grid to any survey
sky fraction via evstats.evs (evs_bin_pdf(..., fsky=...)).

  behroozi : Behroozi+13, Planck15 (original EVS paper, Lovell+23).
  yung     : Yung+24 GUREFT fit; valid z in [6, 19], log10(M/Msun) in [6, 13].

Masses are h-less throughout (see evstats.evs).
"""
import argparse
import os
import multiprocessing
from functools import partial

import numpy as np
import h5py
import hmf
import astropy.cosmology as ac
from astropy.cosmology import FlatLambdaCDM

from evstats import evs

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

# Yung+24 GUREFT cosmology (Planck+20; Om=0.307, H0=67.8, sigma8=0.829, ns=0.960).
GUREFT = (FlatLambdaCDM(H0=67.8, Om0=0.307, Ob0=0.0486, Tcmb0=2.725), 0.829, 0.960)

PRESETS = {
    'behroozi': dict(model='Behroozi', cosmo='Planck15', zlim=(0, 20), mlim=(5, 17)),
    'yung':     dict(model='Yung24',   cosmo='gureft',   zlim=(6, 19), mlim=(6, 13)),
}


def zlim_pdf(zlims, mass_function, mmin, mmax, dm, dz):
    """Worker: whole-sky EVS ingredients (N, f, F, log10m) for one z shell."""
    zmin, zmax = zlims
    N, f, F, log10m = evs._evs_bin(mf=mass_function, zmin=zmin, zmax=zmax,
                                   mmin=mmin, mmax=mmax, dm=dm, dz=dz)
    print(f'  z=[{zmin:.2f},{zmax:.2f}] N={float(N):.3e}', flush=True)
    return N, f, F, log10m


def cosmology(name):
    """(astropy cosmology, sigma_8, n_s); None sigma_8/n_s -> hmf default."""
    if name == 'gureft':
        return GUREFT
    cosmo = getattr(ac, name, None)
    if not isinstance(cosmo, ac.Cosmology):
        raise SystemExit(f"unknown cosmology '{name}'")
    return cosmo, None, None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='behroozi',
                    help='preset (behroozi/yung) or any hmf model name')
    ap.add_argument('--cosmology', help='astropy name or gureft (default: preset)')
    ap.add_argument('-o', '--output', help='output filename in data/')
    ap.add_argument('--dz-bin', type=float, default=0.2, help='shell width')
    ap.add_argument('--dz', type=float, default=0.02, help='z sub-integration step')
    ap.add_argument('--dm', type=float, default=0.02, help='log10 mass step')
    ap.add_argument('--mmin', type=float, help='override log10 min mass (may extrapolate)')
    ap.add_argument('--mmax', type=float, help='override log10 max mass (may extrapolate)')
    ap.add_argument('--zmin', type=float, help='override min redshift (may extrapolate)')
    ap.add_argument('--zmax', type=float, help='override max redshift (may extrapolate)')
    ap.add_argument('--eh-extrap', type=lambda s: s.lower() == 'true', default=None,
                    help='transfer extrapolate_with_eh (true/false); default hmf')
    ap.add_argument('--procs', type=int, default=1,
                    help='parallel processes (default: 1, serial)')
    args = ap.parse_args()

    p = PRESETS.get(args.model.lower(),
                    dict(model=args.model, cosmo='Planck15', zlim=(0, 20), mlim=(5, 17)))
    cosmo, sigma8, ns = cosmology(args.cosmology or p['cosmo'])
    (zmin, zmax), (mmin, mmax) = p['zlim'], p['mlim']
    mmin = args.mmin if args.mmin is not None else mmin
    mmax = args.mmax if args.mmax is not None else mmax
    zmin = args.zmin if args.zmin is not None else zmin
    zmax = args.zmax if args.zmax is not None else zmax

    # linspace keeps edges exactly on [zmin, zmax] (arange can overshoot zmax)
    edges = np.linspace(zmin, zmax, int(round((zmax - zmin) / args.dz_bin)) + 1)
    zlo, zhi = edges[:-1], edges[1:]
    zc = 0.5 * (zlo + zhi)

    # Build inside the valid z range (Yung24's z=0 default would raise).
    kw = dict(hmf_model=p['model'], cosmo_model=cosmo)
    if sigma8 is not None:
        kw['sigma_8'] = sigma8
    if ns is not None:
        kw['n'] = ns
    if args.eh_extrap is not None:
        kw['transfer_params'] = {'extrapolate_with_eh': args.eh_extrap}
    mf = hmf.MassFunction(z=zmin, **kw)

    print(f"{p['model']} / {args.cosmology or p['cosmo']}: {len(zc)} shells, "
          f"z=[{zmin},{zmax}], log10m=[{mmin},{mmax}], fsky=1", flush=True)

    worker = partial(zlim_pdf, mass_function=mf,
                     mmin=mmin, mmax=mmax, dm=args.dm, dz=args.dz)
    if args.procs > 1:
        with multiprocessing.Pool(args.procs) as pool:
            results = pool.map(worker, zip(zlo, zhi))
    else:
        results = [worker(z) for z in zip(zlo, zhi)]

    os.makedirs(DATA, exist_ok=True)
    out = os.path.join(DATA, args.output or f"evs_lightcone_{p['model'].lower()}.h5")
    with h5py.File(out, 'w') as hf:
        hf.create_dataset('log10m', data=results[0][3])
        hf.create_dataset('z', data=zc)
        hf.create_dataset('N', data=np.vstack([r[0] for r in results]))
        hf.create_dataset('f', data=np.vstack([r[1] for r in results]))
        hf.create_dataset('F', data=np.vstack([r[2] for r in results]))
        hf.attrs.update(model=p['model'], cosmology=args.cosmology or p['cosmo'],
                        fsky=1.0, z_range=[zmin, zmax], mass_range=[mmin, mmax],
                        extrapolate_with_eh=str(args.eh_extrap))
    print('wrote', out, flush=True)


if __name__ == '__main__':
    main()
