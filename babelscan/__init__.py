"""
BabelScan
A generic class for reading many types of scan data that you don't need to put in your ear.

requirements: numpy, h5py, imageio
Optional requirements: matplotlib, lmfit

Installation:
  pip install git+https://github.com/DanPorter/babelscan.git

Usage:
  from babelscan import file_loader
  scan1 = file_loader('some/file.nxs', **options)  # creates Scan class
  scan2 = file_loader('another/file.dat', **options)
  scans = scan1 + scan2  # creates MultiScan class

  # Folder monitor:
  from babelscan import FolderMonitor
  mon = FolderMonitor('/some/folder', **options)
  scan = mon.scan(0)  # creates scan from latest file in folder

  # intrument configuration file
  from babelscan import instrument_from_config
  i16 = instrument_from_config('config_files/i16.config')
  experiment = i16.experiment('/data/folder')
  scan = experiment.scan(0)

Scan class
  Scan class contains an internal namespace where each dataset can contain multiple names. Calling the instantiated
  class searches the namespace (and then the file) for a dataset that matches the name given.
    output = scan('name')
  Certain names are reserved and will automatically search the namespace and file for default values:
    scan('axes')  # returns default scanned dataset (x-axis)
    scan('signal')  # returns default signal dataset (y-axis)
  Operations will be evaluated*:
    scan('name1 + name2')  # returns the result of the operation
  For scans with detector images, regions of interest can be generated:
    scan('nroi[127,127,31,31]')  # creates region of interest on image and returs array of sum of roi at each point

  Functions
    scan.title()  # returns str title of scan
    scan.label()  # returns short scan identifier
    scan.scan_command()  # returns definition of the scan command (if available)
    scan.value('name')  # always returns a single value
    scan.string('name')  # returns 'name = value' string
    scan.array(['name1', 'name2'], array_length)  # returns square array of several datasets
    scan.image(idx)  # if available, returns detector image
    scan.string_format('format {name:fmt}')  # returns string formated from namespace
    scan.get_plot_data(xname, yname)  # return data for plotting with errors and labels
    scan.plot(xname, yname)  # create plot**
    scan.fit(xname, yname)  # fit data to peak***

MultiScan class
  MultiScan class is a holder for multiple scans, allowing operations to be performed on all scans in the class.
    scans = MultiScan([scan1, scan2], **options)
    scans = scan1 + scan2
    scans = babelscan.load_files(['file1', 'file2'], **options)
  Works in the same way as underlying scan class - calling the class will return a list of datasets from the scans.
    [output1, output2] = scans('name')

  Functions
    scans.add_variable('name')  # add default parameter that changes between scans, displayed in print(scans)
    scans.array('name')  # return 2D array of scan data
    scans.griddata('name')  # generate 2D square grid of single values for each scan

FolderMonitor class
  FolderMonitor watches a folder (or several) and allows easy loading of files by scan number:
    fm = FolderMonitor('/some/folder', filename_format='%d.nxs')
    scan = fm.scan(12345) # loads '/some/folder/12345.nxs'
    scan = fm.scan(0)  # loads most recent file
    scans = fm.scans(range(12340, 12345))  # MultiScan of several files

  Functions:
    fm.allscanfiles()  # return list of all scan files in folder(s)
    fm.allscannumbers()  # list of all scan numbers in folder(s)
    fm.updating_scan(12345)  # the resulting Scan class will reload on each operation

*functions using eval only available when the "EVAL_MODE" setting is active.
**functions using plot only available if "MATPLOTLIB_PLOTTING" setting is active and matplotlib installed
***functions using fitting only available if lmfit installed

By Dan Porter, PhD
Diamond
2021

Version 0.5.0
Last updated: 28/05/21

Version History:
13/04/21 0.1.0  Version History started.
16/04/21 0.2.0  Added instrument and other additions to Scan, changed container.py to folder_monitor.py
22/04/21 0.3.0  Changed _get_data search path and added _default_values dict, added volume.py, settings.py
26/04/21 0.4.0  Various changes and fixes after testing with i06, i10 files
04/05/21 0.4.1  Added names dict to axes/signal from cmd functions
28/05/21 0.5.0  Tidied up code, various fixes

-----------------------------------------------------------------------------
   Copyright 2021 Diamond Light Source Ltd.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

 Dr Daniel G Porter, dan.porter@diamond.ac.uk
 www.diamond.ac.uk
 Diamond Light Source, Chilton, Didcot, Oxon, OX11 0DE, U.K.
"""


__version__ = "0.5.0"
__date__ = "28/05/2021"


from .__settings__ import EVAL_MODE
# from .babelscan import Scan, MultiScan
# from .hdf import HdfScan
# from .dat import DatScan
# from .csv import CsvScan
from .hdf import HdfWrapper, HdfDataset
from .folder_monitor import create_scan, file_loader, hdf_loader, load_files, find_files, FolderMonitor
from .instrument import Instrument, instrument_from_config
from .functions import save_to_config


def version_info():
    return 'babelscan version %s (%s)' % (__version__, __date__)


def module_info():
    import sys
    out = 'Python version %s' % sys.version
    out += '\n%s' % version_info()
    # Modules
    import numpy
    out += '\n     numpy version: %s' % numpy.__version__
    import h5py
    out += '\n      h5py version: %s' % h5py.__version__
    import imageio
    out += '\n   imageio version: %s' % imageio.__version__
    try:
        import matplotlib
        out += '\nmatplotlib version: %s' % matplotlib.__version__
    except ImportError:
        out += '\nmatplotlib version: None'
    try:
        import lmfit
        out += '\n     lmfit version: %s' % lmfit.__version__
    except ImportError:
        out += '\n     lmfit version: None'
    import os
    out += '\nRunning in directory: %s\n' % os.path.abspath('.')
    return out
