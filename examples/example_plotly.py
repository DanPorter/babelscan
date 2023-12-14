"""
BabelScan Example
Example using the plotly plotting package
"""

import os
import babelscan
import plotly.graph_objects as go

mydir = os.path.expanduser('~')
scan = babelscan.file_loader(mydir + r"\Dropbox\Python\ExamplePeaks\794940.nxs")

blob = scan.plot.plotly_blob('axes', ['sum', 'roi2_sum'])

fig = go.Figure(blob)
fig.show()

print('Finished')
