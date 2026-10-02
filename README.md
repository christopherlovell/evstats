# evstats
A python module for calculating extreme value statistics of the halo and galaxy stellar mass distributions. Full details provided in [Lovell et al. 2023](https://academic.oup.com/mnras/article/518/2/2511/6823705).

<img src="http://www.christopherlovell.co.uk/assets/img/evs-1400.webp" alt="drawing" width="400"/>

### Installation

Clone this repository, then run the following in your chosen python environment

```
pip install .
```

You can then use evstats as so:

```
from evstats import evs
from evstats import stats
from evstats import stellar
```

The star-formation-history module (`evstats.sfr`) provides
`mass_growth_track` / `plot_mass_growth_track`, which project an observed
stellar mass back in redshift for an assumed (parametric or binned) star
formation history, for comparison against the EVS contours. It has extra
dependencies (`unyt` and `cosmos-synthesizer`). Install them with:

```
pip install ".[sfr]"
```

### An example
A notebook showing a simple example of how to create contours in the stellar mass -- redshift plane, for arbitrary survey areas, is available [here](https://nbviewer.org/github/christopherlovell/evstats/blob/main/example/example.ipynb).

### The paper
All of the plots and analysis in [Lovell et al. 2023](https://academic.oup.com/mnras/article/518/2/2511/6823705) can be recreated using the scripts in `example/paper/`.
