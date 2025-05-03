import sys, os
from functools import partial

import numpy as np
import h5py
import multiprocessing

from astropy.cosmology import Planck15
import astropy.units as u
import hmf

from evstats import evs


def zlim_pdf(zlims, mass_function=hmf.MassFunction(hmf_model='Behroozi'), mmin=4, mmax=17, dm=0.005, dz=0.01):
    """
    individual worker for multiprocessing

    returns phi_max for the zrange specified, and the log masses
    """
    
    zmin = zlims[0]
    zmax = zlims[1]    

    # print(zmin, zmax)
    N, f, F, ln10m_range = evs._evs_bin(mf=mass_function, zmin=zmin, zmax=zmax, 
                                        mmin=mmin, mmax=mmax, dm=dm, dz=dz)

    print(zmin, N, f[0], F[0])    
    return N, f, F, ln10m_range


if __name__ == '__main__':
    
    _out_str = sys.argv[1]
    # _out_str = 'evs_pdf'
    
    if os.path.exists('data/%s'%_out_str):
        # if sb.yesno('File exists. Are you sure you wish to overwrite the existing file?'):
        os.remove('data/%s'%_out_str)
        # else:
        #     print("Exiting...")
        #     sys.exit()

    
    # zlowlim = np.arange(0, 14.9, 0.1)
    # zupplim = np.arange(0.1, 15, 0.1)
    
    zlowlim = np.arange(0, 44.9, 0.5)
    zupplim = np.arange(0.1, 45, 0.5)
    zcentre = zlowlim + np.diff(zlowlim)[0]/2
    
    dz = 0.05  # 0.01
    mmin = 4; mmax = 14; dm=0.03  # 0.01
    
    mass_function = hmf.MassFunction()  # hmf_model='Behroozi')
    mass_function.cosmo_model = Planck15
    
    # NUM_PROCS = multiprocessing.cpu_count()
    NUM_PROCS = 8
    print(NUM_PROCS)
    sys.stdout.flush()

    # pool = multiprocessing.Pool(processes=NUM_PROCS)
    # results = pool.map(partial(zlim_pdf, mass_function=mass_function, 
    #                            mmin=mmin, mmax=mmax, dm=dm, dz=dz), \
    #                    zip(zlowlim, zupplim))

    results = [
        zlim_pdf(
            (_zlowlim, _zupplim),
            mass_function=mass_function,
            mmin=mmin,
            mmax=mmax,
            dm=dm,
            dz=dz
        ) for _zlowlim, _zupplim in zip(zlowlim, zupplim)
    ]
        
    
    N = np.vstack([dat[0] for dat in results])
    f = np.vstack([dat[1] for dat in results])
    F = np.vstack([dat[2] for dat in results])
    log10m = results[0][3]
    
    with h5py.File(f'data/{_out_str}', 'w') as hf:
        hf.create_dataset('log10m', data=log10m)
        hf.create_dataset('z', data=zcentre)
        hf.create_dataset('N', data=N)
        hf.create_dataset('f', data=f)
        hf.create_dataset('F', data=F)
