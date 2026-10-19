# evstats

evstats is a python package for extreme value statistics (EVS) in astrophysics: the probability distribution of the most extreme object, such as the most massive or the brightest, expected in a survey of a given area and redshift range. EVS applies to physical distribution functions, such as the halo and galaxy stellar mass functions, and to observed ones, such as the distribution of galaxy fluxes in a given band. 

The method is described in [Lovell et al. 2023](https://academic.oup.com/mnras/article/518/2/2511/6823705) and [Lovell et al. 2026](https://arxiv.org/abs/2610.23016).

```{image} https://www.christopherlovell.co.uk/assets/img/evs-1400.webp
:width: 400px
```

## Installation

```bash
pip install evstats
```

evstats requires Python 3.12 or newer. The core package needs only numpy, scipy, astropy and [hmf](https://github.com/halomod/hmf). hmf is held at 3.6, since other versions shift the halo mass function and the tabulated EVS in `example/data/` were computed with it.

Optional dependencies are installed as extras:

```bash
pip install "evstats[examples]"     # matplotlib and h5py
pip install "evstats[observables]"  # also Synthesizer and unyt
```

The examples in these pages need the `examples` extra, and making new flux tables needs `observables`. The examples also read tabulated data from the `example/` directory of the repository, which is not part of the pip package; to run them, clone the repository and run the notebooks from `docs/`:

```bash
git clone https://github.com/christopherlovell/evstats
cd evstats
pip install ".[examples]"
```

## Overview

1. [Halo mass EVS](halo.ipynb): the PDF of the most massive halo in a redshift shell and survey area, from the halo mass function.
2. [Stellar mass EVS](stellar.ipynb): the PDF of the most massive galaxy's stellar mass, from the baryon fraction and stellar fraction distribution.
3. [Flux EVS](observables.ipynb): the flux of the brightest galaxy, for a given forward model for emission.

All of the plots and analysis in Lovell et al. 2023 can be recreated with the scripts in `example/paper/`. Plots and analysis from Lovell et al. 2026 can be recreated from the scripts in `example/obserables`.

## Citing

If you use evstats, please cite [Lovell et al. 2023](https://academic.oup.com/mnras/article/518/2/2511/6823705). If you use the observables functionality, please additionally cite [Lovell et al. 2026](https://arxiv.org/abs/2610.23016).

```{toctree}
:hidden:

halo
stellar
observables
api
```
