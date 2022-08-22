import numpy as np
import h5py
import matplotlib.pyplot as plt

from astropy.cosmology import Planck15
import astropy.units as u
from astropy.table import Table

from evstats import evs
from evstats.stats import compute_conf_ints, eddington_bias
from evstats.stellar import apply_fs_distribution


with h5py.File('../data/evs_all.h5','r') as hf:
    log10m = hf['log10m'][:]
    f = hf['f'][:]
    F = hf['F'][:]
    N = hf['N'][:]
    z = hf['z'][:]


survey_areas = [
                236 * u.arcmin**2, # JADES
                0.6 * u.degree**2, # COSMOS Webb
                20 * u.degree**2,  # Euclid Deep Field North
                1700 * u.degree**2,  # Roman High Latitude Wide Area Survey
                15000 * u.degree**2,  # Euclid Wide Survey
                (41252.96 * u.degree**2).to(u.arcmin**2)  # whole sky
               ]

survey_area_short = [
                '$236 \; \mathrm{arcmin}^2$',
                '$0.6 \; \mathrm{degree}^2$',
                '$20 \; \mathrm{degree}^2$',
                '$1700 \; \mathrm{degree}^2$',
                '$15000 \; \mathrm{degree}^2$',
                '$41252.96 \; \mathrm{degree}^2$'
            ]

survey_names = ['JADES',
                'COSMOS Web',
                'Euclid Deep Field (N)',
                'Roman HLWAS',
                'Euclid Wide',
                'Whole Sky']

survey_names_short = ['JADES',
                'COSMOS_Web',
                'Euclid_DF_N',
                'Roman_HLWAS',
                'Euclid_Wide',
                'Whole_Sky']

# whole_sky = (41252.96 * u.deg**2).to(u.arcmin**2)
whole_sky = 41252.96 * u.degree**2
f_b = 0.16

fig, axes = plt.subplots(3, 2, figsize=(10.5, 13.9))
plt.subplots_adjust(hspace=0.1, wspace = 0.1)
colors = ['brown','lightcoral','mistyrose']# ['steelblue','lightskyblue','powderblue'] 

for ax, survey_area, survey_name, name_short, area_short in \
                zip(axes.flatten(), survey_areas, survey_names, survey_names_short, survey_area_short):
    
    fsky = float((survey_area / whole_sky).to(''))
    print(survey_name, survey_area, fsky) 
    phi_max = evs._apply_fsky(N, f, F, fsky)

    CI_mhalo = compute_conf_ints(phi_max, log10m)
    CI_baryon = np.log10(10**CI_mhalo * f_b)

    mstar_pdf = np.vstack([apply_fs_distribution(_phi_max, log10m, _N=int(1e5), method='lognormal') \
            for _phi_max in phi_max])
    CIs = compute_conf_ints(mstar_pdf, log10m)
    
    ## save as ECSV
    for _ci, _type in zip([CIs, CI_mhalo], ['stellar', 'halo']):
        data = Table()
        data['z'] = z
        data['-3 sigma'] = CIs[:,0]
        data['-2 sigma'] = CIs[:,1]
        data['-1 sigma'] = CIs[:,2]
        data['median'] = CIs[:,3]
        data['+1 sigma'] = CIs[:,4]
        data['+2 sigma'] = CIs[:,5]
        data['+3 sigma'] = CIs[:,6]
        data.write('../data/%s_%s.ecsv' % (name_short, _type), overwrite=True)

    ax.fill_between(z, CIs[:,0], CIs[:,6], alpha=1, color=colors[0])
    ax.fill_between(z, CIs[:,1], CIs[:,5], alpha=1, color=colors[1])
    ax.fill_between(z, CIs[:,2], CIs[:,4], alpha=1, color=colors[2])
    ax.plot(z, CIs[:,3], linestyle='dotted', c='black')
    ax.plot(z, CI_baryon[:,6], linestyle='dashed', color='black')

    ax.set_xlim(2,19.7); ax.set_ylim(6.1,14.5)
    # ax.text(0.05, 0.04, '$A = 850.7 \; \mathrm{arcmin}^2$', size=12, color='black', alpha=0.8, transform = ax.transAxes)
    ax.text(0.05, 0.04, survey_name, size=17, color='black', alpha=0.5, transform = ax.transAxes)
    ax.text(0.05, 0.12, '$A = $%s'%area_short, size=12, color='black', alpha=0.8, transform = ax.transAxes)
    
    line1 = plt.Line2D((0,1),(0,0), color=colors[0], linewidth=5)
    line2 = plt.Line2D((0,1),(0,0), color=colors[1], linewidth=5)
    line3 = plt.Line2D((0,1),(0,0), color=colors[2], linewidth=5)
    line4 = plt.Line2D((0,1),(0,0), color='black', linestyle='dotted', linewidth=2)
    line5 = plt.Line2D((0,1),(0,0), color='black', linestyle='dashed', linewidth=2)
    line_dummy = plt.Line2D((0,1),(0,0), color='white')
    leg = ax.legend(handles=[line4,line5,line_dummy,line3,line2,line1], 
               labels=['$\mathrm{med}(M^{\star}_{\mathrm{max}})$','$f_{\mathrm{b}}$ $+3\sigma$','',
                   '$1\sigma$', '$2\sigma$', '$3\sigma$'],
                    frameon=False, loc='upper right', fontsize=12, ncol=2)

    vp = leg._legend_box._children[-1]._children[0] 
    for c in vp._children: c._children.reverse() 
    vp.align="right" 

    ax.grid(alpha=0.3, color='white')

    
for ax in axes[-1, :]:
    ax.set_xlabel('$z$', size=17)

for ax in axes[:, 0]:
    ax.set_ylabel('$\mathrm{log_{10}}(M^{\star}_{\mathrm{max}} \,/\, M_{\odot})$', size=15)
    
# plt.show()
_out_str = 'surveys'
plt.savefig('plots/evs_%s.pdf'%_out_str, bbox_inches='tight', dpi=200); plt.close()