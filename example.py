"""
BabelScan example script
"""

import os
import numpy as np
import matplotlib.pyplot as plt


from babelscan import file_loader, FolderMonitor


datadirs = [
    r"C:\Users\dgpor\OneDrive - Diamond Light Source Ltd\I16\Nexus_Format\example_nexus",
    r"C:\Users\dgpor\Dropbox\Python\ExamplePeaks"
]


exp = FolderMonitor(datadirs)
scan = exp.scan(877619)
print(scan)

scan.plot()

scan.plot.plot_image()


