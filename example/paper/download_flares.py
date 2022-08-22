import flares
fl = flares.flares(fname='/cosma7/data/dp004/dc-love2/codes/flares/data/flares.h5')

fl_mstar = fl.load_dataset('Mstar',arr_type='Galaxy')
fl_m200 = fl.load_dataset('M200',arr_type='Galaxy')

mstar_max = [None for tag in fl.tags]
m200_max = [None for tag in fl.tags]
for i,tag in enumerate(fl.tags):
    mstar_max[i] = np.hstack([np.log10(fl_mstar[reg][tag].max() * 1e10) for reg in fl.halos \
                                if len(fl_mstar[reg][tag]) > 0]).max()
    m200_max[i] = np.hstack([np.log10(fl_m200[reg][tag].max() * 1e10) for reg in fl.halos \
                                if len(fl_m200[reg][tag]) > 0]).max()


np.savetxt('data/flares_m200_max.txt',[fl.zeds,m200_max])
np.savetxt('data/flares_mstar_max.txt',[fl.zeds,mstar_max])
