"""
BabelScan example script
Load a series of scan files, fit the peak in each scan and plot against a changing variable,
in this case scanning accross the surface of a sample in sx and sy
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import babelscan

mydir = os.path.expanduser('~')
instrument = babelscan.instrument_from_config('../config_files/i16.config')
exp = instrument.experiment(mydir + r"\OneDrive - Diamond Light Source Ltd\I16\Nexus_Format\example_nexus")

scan_numbers = range(794932, 794947, 1)
scans = exp.scans(scan_numbers, ['sperp', 'spara'])

print(scans)

scans.fit.multi_peak_fit(peak_distance_idx=5, print_result=True, plot_result=True)
scans.plot.plot_simple('sperp', 'amplitude')
scans.plot.multiplot(yaxis=['signal', 'yfit'])
scans.plot.show()
