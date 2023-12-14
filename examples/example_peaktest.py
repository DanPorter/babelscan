"""
BabelScan example script
Test peaks
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import babelscan
from babelscan.fitting import error_func

mydir = os.path.expanduser('~')
examplepeaks = mydir + r"\Dropbox\Python\ExamplePeaks\%d.dat"
examplenexus = mydir + r"\OneDrive - Diamond Light Source Ltd\I16\Nexus_Format\example_nexus\%d.nxs"

# \Python\ExamplePeaks
strong_peaks = [387091, 387270, 387276, 393793, 393874, 393973]
weak_peaks = [387282, 387283, 387295, 387301, 393922, 393931, 393940, 393976, 393977, 571654]
non_peaks = [387288, 387294, 394683, 394687, 394691, 394695, 571649]
files = []
files += [examplepeaks % s for s in strong_peaks]
files += [examplepeaks % s for s in weak_peaks]
files += [examplepeaks % s for s in non_peaks]

# \Nexus_Format\example_nexus
scan_numbers = range(794932, 794947, 1)
files += [examplenexus % s for s in scan_numbers]

tests=[794941, 571649, 393977]

# for file in files:
#     scan = babelscan.file_loader(file)
#     scan.save_plot_data(error_op=error_func)

out = []
for file in files:
    scan = babelscan.file_loader(file)
    results = scan.fit.multi_peak_fit(print_result=False, plot_result=True)
    print(results)
plt.show()
