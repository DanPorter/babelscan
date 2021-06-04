"""
BabelScan example script
Load a series of scan files, fit the peak in each scan and plot against a changing variable,
in this case scanning accross the surface of a sample in sx and sy
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import babelscan

instrument = babelscan.instrument_from_config('../config_files/i16.config')
exp = instrument.experiment(r"C:\Users\dgpor\OneDrive - Diamond Light Source Ltd\I16\Nexus_Format\example_nexus")

scan_numbers = range(794932, 794947, 1)
scans = exp.scans(scan_numbers, ['sperp', 'spara'])

scans.fit()

scans.plot.plot_simple('sperp', 'amplitude')

scans.plot.multiplot(yaxis=['signal', 'fit'])
