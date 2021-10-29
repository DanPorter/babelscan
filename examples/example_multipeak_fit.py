"""
Bablescan examples
Fit multiple peaks in single scan
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import babelscan
from babelscan.fitting import multipeakfit

scan = babelscan.file_loader(r"C:\Users\dgpor\Dropbox\Python\ExamplePeaks\794940.nxs")

print(scan)

x, y, dy, xlabel, ylabel = scan.get_plot_data()

# res = multipeakfit(x, y, np.sqrt(np.abs(y)+1), npeaks=None, print_result=True, plot_result=True)
res = scan.fit.multi_peak_fit(print_result=True, plot_result=True)
plt.show()

print('Finished')
