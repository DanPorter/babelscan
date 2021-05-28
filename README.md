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

### Usage
```python
# Python script
import babelscan
scan1 = babelscan.file_loader('12345.nxs')
scan2 = babelscan.file_loader('i16_1234.dat')
exp = babelscan.FolderMonitor('/path/to/files')
scan3 = exp.scan(0)  # returns latest scan in directory
scans = scan1 + scan2 + scan3  # creates MultiScan object that combines the 3 datasets
```

### Installation
#### requirements: numpy, h5py, imageio, [optional: matplotlib, lmfit]
#### available from: https://github.com/DanPorter/babelscan
```commandline
pip install git+https://github.com/DanPorter/babelscan.git
```

### Examples
```python
import babelscan
scan = babelscan.file_loader('12345.nxs')

print(scan)  # prints scan information

x = scan('axes')  # finds default xaxis in Nexus files
y = scan('signal')  # finds default yaxis in Nexus files

title = scan.title()  # generates a plot title

im = scan.image(0)  # returns first detector image if scan contains 3D data
```
See the included example_*.py files for more examples.