# BabelScan
BabelScan is a format independent data structure for holding different types of data from a scan file.

Various file formats are supported, including Nexus and HDF formats, as well as older ASCII file formats. 
BabelScan is implicitly Lazy Loading and only loads data when requested, making it fast and lightweight.

Data fields are stored in a format independent way, with the abiliity to find close matches to a 
requested field.

3D datasets such as images from detectors are specially handelled and contain several special functions.

If packages lmfit and matplotlib are available, fitting and plotting options are included in the BabelScan object.

A FolderMonitor class allows loading of specific files in numeric order

An Instrument class holds specific configuration for generation of bespoke BabelScan objects.

You can read the documentation [here!](https://babelscan.readthedocs.io/en/latest/) 

[![Documentation Status](https://readthedocs.org/projects/babelscan/badge/?version=latest)](https://babelscan.readthedocs.io/en/latest/?badge=latest)

By Dan Porter, Diamond Light Source Ltd. 2023

### Usage
```python
# Python script
import babelscan
scan1 = babelscan.file_loader('12345.nxs')
scan2 = babelscan.file_loader('i16_1234.dat')
exp = babelscan.FolderMonitor('/path/to/files')
scan3 = exp.scan(0)  # returns latest scan in directory
scans = scan1 + scan2 + scan3  # creates MultiScan object that combines the 3 datasets

# Folder monitor:
mon = babelscan.FolderMonitor('/some/folder', **options)
scan = mon.scan(0)  # creates scan from latest file in folder

# intrument configuration file
i16 = babelscan.instrument_from_config('config_files/i16.config')
experiment = i16.experiment('/data/folder')
scan = experiment.scan(12345)
print(scan)  # displays I16 metadata by default
```

### Installation
**requirements:** *numpy, h5py, imageio, python-dateutil*, [**optional:** *matplotlib, lmfit*]

**available from: https://github.com/DanPorter/babelscan**

Latest version from github:
```commandline
pip install git+https://github.com/DanPorter/babelscan.git
```
Stable version from [PyPI](https://pypi.org/project/babelscan/):
```commandline
pip install babelscan
```


### Examples

```python
import babelscan

scan = babelscan.file_loader('12345.nxs')

print(scan)  # prints scan information

en = scan('energy')  # finds data field named 'energy', returns data
val = scan('sum/Transmission')  # Finds 'sum' and 'Transmission' fields, evaluates result

x = scan('axes')  # finds default xaxis in Nexus files
y = scan('signal')  # finds default yaxis in Nexus files

title = scan.title()  # generates a plot title

im = scan.image(0)  # returns first detector image if scan contains 3D data

# Automatically generate x, y, error data and labels for plotting
# Here we also show off the automatic "region of interest" specification,
# as well as the automatic normalisation and error generation
x, y, dy, xlab, ylab = scan.get_plot_data('axes', 'nroi_peak[31,31]', '/count_time/Transmission', 'np.sqrt(x+0.1)')

# If matplotlib is installed, plotting behaviour is enabled:
scan.plot()  # creates a figure and plots the default axes
scan.plot.plotline('axes', 'signal', 'b-')  # command similar to plt.plot
scan.plot.image(index)  # creates figure and plots detector image

# If lmfit is installed, fitting behaviour is enabled:
scan.fit()  # fits a gaussian peak to the default axes
scan.fit.fit('axes', 'signal')  # Fits a gaussian peak to choosen axes
scan.fit.multi_peak_fit('axes', 'signal')  # Automatically runs a peak search and fits multiple peaks
# The resulting parameters are stored in the namespace:
scan('amplitude, stderr_amplitude')
```
See the included example_*.py files for more examples.