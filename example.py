"""
BabelScan example script
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from babelscan import file_loader, FolderMonitor

mydir = os.path.expanduser('~')
datadirs = [
    mydir + r"\OneDrive - Diamond Light Source Ltd\I16\Nexus_Format\example_nexus",
    mydir + r"\Dropbox\Python\ExamplePeaks"
]

exp = FolderMonitor(datadirs)
scan = exp.scan(877619)
print(scan)

scan.plot()

scan.plot.image()
scan.plot.show()

