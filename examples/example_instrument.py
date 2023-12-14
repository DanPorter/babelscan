"""
BabelScan Example
"""

import os
import babelscan

# Instrument (holds configuration settings)
i16 = babelscan.instrument_from_config('../config_files/i16.config')

# Experiment (watches folder or folders for scan files)
mydir = os.path.expanduser('~')
exp = i16.experiment(mydir + r"\OneDrive - Diamond Light Source Ltd\I16\Nexus_Format\example_nexus")

# Scan - holds single scan data
scan = exp(0)

print(scan)

