"""
BabelScan Example
Example using the plotly plotting package
"""

import babelscan
import plotly.graph_objects as go

scan = babelscan.file_loader(r"C:\Users\dgpor\Dropbox\Python\ExamplePeaks\794940.nxs")

blob = scan.plot.plotly_blob('axes', ['sum', 'roi2_sum'])

fig = go.Figure(blob)
fig.show()

print('Finished')