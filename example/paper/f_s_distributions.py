import numpy as np
from scipy.stats import uniform,norm,expon,truncnorm,lognorm,gaussian_kde
import matplotlib.pyplot as plt

from evstats import evs

## params for truncated log norm
myclip_a = 0; myclip_b = 1; my_mean = 0.2; my_std = 0.1
_a, _b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std

## plot f_s distributions
cmap = plt.get_cmap('Spectral', 4)
fig,ax=plt.subplots(1,1,figsize=(4.5,3.8))
_x = np.linspace(0,1,300)
ax.plot(_x, truncnorm.pdf(_x, _a, _b, loc = my_mean, scale = my_std), label='Truncated Normal', color=cmap(0), lw=2)
ax.plot(_x, uniform.pdf(_x, loc=0, scale=1), label='Uniform', color=cmap(1), lw=2)  # U[0,1]
ax.plot(_x, expon.pdf(_x, scale=0.1), label='Exponential', color=cmap(2), lw=2)
ax.plot(_x, lognorm.pdf(_x, s=1, scale=np.exp(-2)), label='Truncated Log-Normal', lw=4, color=cmap(3))
ax.legend(frameon=False)
ax.set_xlim(0,1); plt.ylim(0,)
ax.set_xlabel('$f_{\star}$')
ax.set_ylabel('PDF ($f_{\star}$)')

plt.show()
# plt.savefig('plots/fs_distribution.pdf', bbox_inches='tight'); plt.close()