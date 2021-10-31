"""
BabelScan example script
Load a series of scan files and create a html page of the details and plots
Creates a Folder called "example_scans" in the user home directory
"""

import os
import babelscan

instrument = babelscan.instrument_from_config('../config_files/i16.config')
exp = instrument.experiment(r"C:\Users\dgpor\OneDrive - Diamond Light Source Ltd\I16\Nexus_Format\example_nexus")

scan_numbers = range(794932, 794947, 1)
scans = exp.scans(scan_numbers, ['sperp', 'spara'])

folder_name = os.path.expanduser('~/example_scans')
scans.plot.plot_details_to_html(folder_name)

print('Finished!')
