import numpy as np
import eagle_IO.eagle_IO as E

## read in EAGLE data
nThreads = 4
sim = '/cosma7/data//Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data'
tags = np.array(['000_z020p000', '001_z015p132', '002_z009p993', '003_z008p988', 
                 '004_z008p075', '005_z007p050', '006_z005p971', '007_z005p487', 
                 '008_z005p037', '009_z004p485', '010_z003p984', '011_z003p528', 
                 '012_z003p017', '013_z002p478', '014_z002p237', '015_z002p012', 
                 '016_z001p737', '017_z001p487', '018_z001p259', '019_z001p004', 
                 '020_z000p865', '021_z000p736', '022_z000p615', '023_z000p503', 
                 '024_z000p366', '025_z000p271', '026_z000p183', '027_z000p101', '028_z000p000'])

_zeds = [float(tag[5:].replace('p','.')) for tag in tags]

m_max = [None for zed in _zeds]
mstar_max = [None for zed in _zeds]
for i,tag in enumerate(tags):
    m = E.read_array('SUBFIND', sim, tag, 'Subhalo/Stars/Mass', 
                     noH=True, physicalUnits=True, numThreads=nThreads)
    mstar_max[i] = np.log10(m.max() * 1e10)

    # m = E.read_array('SUBFIND', sim, tag, 'Subhalo/Mass', 
    m = E.read_array('SUBFIND_GROUP', sim, tag, 'FOF/Group_M_Crit200', 
                     noH=True, physicalUnits=True, numThreads=nThreads)
    m_max[i] = np.log10(m.max() * 1e10)


np.savetxt('data/eagle_m200_max.txt',[_zeds,m_max])
np.savetxt('data/eagle_mstar_max.txt',[_zeds,mstar_max])

_zeds,mstar_max = np.loadtxt('data/eagle_mstar_max.txt')
_zeds,m_max = np.loadtxt('data/eagle_m200_max.txt')

plt.scatter(_zeds, m_max, marker='o', s=10, color='orange')

plt.xlim(0,21)
plt.xlabel('$z$')
plt.ylabel('$M^{\mathrm{200}}_{\mathrm{max}} \,/\, \mathrm{M_{\odot}}$')
plt.show()