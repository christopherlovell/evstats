# evstats
A python package for extreme value statistics (EVS) in astrophysics: the probability distribution of the most extreme object, such as the most massive or the brightest, expected in a survey of a given area and redshift range. EVS applies to physical distribution functions, such as the halo and galaxy stellar mass functions, and to observed ones, such as the distribution of galaxy fluxes in a given band. An observed object far beyond this distribution is in tension with the model behind it. The method is described in [Lovell et al. 2023](https://academic.oup.com/mnras/article/518/2/2511/6823705).

<img src="https://www.christopherlovell.co.uk/assets/img/evs-1400.webp" alt="drawing" width="400"/>

### Installation

```
pip install evstats
```

evstats requires Python 3.12 or newer. Optional dependencies are installed as extras:

```
pip install "evstats[examples]"     # matplotlib and h5py, for plotting and the examples
pip install "evstats[observables]"  # also Synthesizer and unyt, for making new flux tables
```

### Documentation
Worked examples for the halo mass, stellar mass and flux distributions, and the API reference, are at https://www.christopherlovell.co.uk/evstats/.

### The paper
All of the plots and analysis in [Lovell et al. 2023](https://academic.oup.com/mnras/article/518/2/2511/6823705) can be recreated using the scripts in `example/paper/`.
