# Farthest sampler

Some code to perform FPS selections of features or samples.

## Installation

```bash
pip install git+https://github.com/Luthaf/farthest-sampler
```

## Example

```py
import numpy as np
from farthest_sampler import select_fps_voronoi, select_fps_standard

points = np.load("my-file.npy")    # or generate the data dynamically

assert len(points.shape) == 2      # only 2D array
assert points.dtype == np.float64  # only array of float64 values

selected_voronoi = select_fps_voronoi(
    points,
    300, # number of points to select
    0, # first selected point
)

selected_standard = select_fps_standard(
    points,
    300, # number of points to select
    0, # first selected point
)

assert np.all(selected_voronoi == selected_standard)

# selected_{voronoi,standard} contain the indexes of the selected points
```

## Performance

Here are the result of the benchmarks included in this repository on the
[Boston](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html#sklearn.datasets.load_boston)
house price dataset (shape 506x13), and a dataset of SOAP features (shape
2520x1362).

```
group                         main/Standard FPS/       main/Voronoi FPS/
-----                         ------------------       -----------------
100 samples/Boston dataset    1.00      3.4±0.51ms     1.17      4.0±0.78ms
200 samples/Boston dataset    1.00      7.0±0.45ms     1.25      8.8±1.88ms
100 samples/SOAP              1.61     78.3±7.48ms     1.00     48.6±4.12ms
200 samples/SOAP              1.91    155.6±9.72ms     1.00     81.5±4.38ms
10 features/Boston dataset    1.00   171.6±28.89µs     1.41   242.5±41.00µs
100 features/SOAP             1.00     75.8±7.63ms     1.21     92.0±3.14ms
200 features/SOAP             1.00   153.6±18.42ms     1.17   179.2±35.07ms
```

Overall, VoronoiFPS is faster than the standard full algorithm when selecting
samples in a reasonably uniform space (i.e. SOAP feature space, but not the
Boston dataset).

To run these benchmarks for yourself, you'll need
[critcmp](https://github.com/BurntSushi/critcmp):

```bash
cargo bench -- --save-baseline main
critcmp main -g ".*FPS/(.*)"
```
