"""
BabelScan Example
"""

import os
import babelscan

mydir = os.path.expanduser('~')
scan = babelscan.file_loader(mydir + r"\Dropbox\Python\ExamplePeaks\810002.nxs")
print(scan)

# HDF options
print(scan.find_address('energy'))  # find addresses that match name "energy"
print(scan.address('energy'))  # display address that is choosen by default

# Load HDF file (wrapper around h5py.File)
hdf = scan.load_hdf()
print(hdf.tree(detail=True))

# Reloading dataset
dataset = scan.dataset('sum')
print(dataset)

print(scan('/entry1/sample/ub_matrix'))
print(scan('azih, azik, azil'))
print(scan('h, k, l'))

