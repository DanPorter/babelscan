"""
Bablescan examples
Fit multiple peaks in single scan
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import babelscan

mydir = os.path.expanduser('~')
scan = babelscan.file_loader(mydir + r"\Dropbox\Python\ExamplePeaks\794940.nxs")
scan.set_error_operation()
print(scan)

res = scan.fit.multi_peak_fit(print_result=True, plot_result=True)
plt.show()

print('Finished')
