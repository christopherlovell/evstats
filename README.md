# evstats
A python module for calculating extreme value statistics of the halo and galaxy stellar mass distributions. Full details provided in [Lovell et al. 2023](https://academic.oup.com/mnras/article/518/2/2511/6823705).

<img src="https://www.christopherlovell.co.uk/images/jwst_evs.png" alt="drawing" width="400"/>

### Installation

Clone this repository, then run the following in your chosen python environment

```
python setup.py install
```

You can then use evstats as so:

```
from evstats import evs
from evstats import stats
from evstats import stellar
```

### An example
A notebook showing a simple example of how to create contours in the stellar mass -- redshift plane, for arbitrary survey areas, is available [here](https://nbviewer.org/github/christopherlovell/evstats/blob/main/example/example.ipynb).

### The paper
All of the plots and analysis in [Lovell et al. 2023](https://academic.oup.com/mnras/article/518/2/2511/6823705) can be recreated using the scripts in `example/paper/`.
