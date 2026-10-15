# evstats
A python module for calculating extreme value statistics of the halo and galaxy stellar mass distributions. Full details provided in [Lovell et al. 2023](https://academic.oup.com/mnras/article/518/2/2511/6823705).

<img src="http://www.christopherlovell.co.uk/assets/img/evs-1400.webp" alt="drawing" width="400"/>

### Installation

```
pip install evstats
```

or, from a clone of this repository, `pip install .`. You can then use evstats as so:

```
from evstats import evs
from evstats import stats
from evstats import stellar
```

The core package needs only numpy, scipy, astropy and [hmf](https://github.com/halomod/hmf). hmf is held below 3.7, whose changed defaults shift the halo mass function; the tabulated EVS in `example/data/` were computed with hmf 3.6.

Optional dependencies are installed as extras:

```
pip install "evstats[examples]"     # matplotlib and h5py: plotting, and the scripts in example/
pip install "evstats[observables]"  # also Synthesizer and unyt: the photometry in example/observables/
```

The star-formation-history module (`evstats.sfr`) provides
`mass_growth_track` / `plot_mass_growth_track`, which project an observed
stellar mass back in redshift for an assumed (parametric or binned) star
formation history, for comparison against the EVS contours. Plotting needs
the `examples` extra, and `average_sfr_over_window` needs the `observables` extra.

### An example
A notebook showing a simple example of how to create contours in the stellar mass -- redshift plane, for arbitrary survey areas, is available [here](https://nbviewer.org/github/christopherlovell/evstats/blob/main/example/example.ipynb).

### Observables
The scripts in `example/observables/` turn the stellar mass EVS into JWST fluxes using [Synthesizer](https://github.com/synthesizer-project/synthesizer), so need the `observables` extra. They read flux tables that are not distributed with this repository; to make them:

1. Get the Synthesizer grid for the fiducial model, `bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-c23.01-sps`, and set `GRID_DIR` in `evs_file_generator.py` to the directory containing it.
2. From `example/observables/`, generate the flux tables for every star formation history with that grid, at the fiducial metallicity:
   ```
   python evs_file_generator.py "" bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy 0.01
   ```
3. Plot the EVS flux predictions for each star formation history:
   ```
   python evs_sfh.py
   ```

Running `evs_file_generator.py` with no arguments makes the tables for every SFH, SPS model and metallicity, which the other scripts need; this is slow.

### The paper
All of the plots and analysis in [Lovell et al. 2023](https://academic.oup.com/mnras/article/518/2/2511/6823705) can be recreated using the scripts in `example/paper/`.
