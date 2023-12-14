"""
BabelScan Example
List directory scan contents
"""

import os
import babelscan

mydir = os.path.expanduser('~')
folder = babelscan.FolderMonitor(mydir + r"\OneDrive - Diamond Light Source Ltd\I16\Nexus_Format\example_nexus")

all_scan_files = folder.allscanfiles()  # list of str file names

all_scan_numbers = folder.allscannumbers()  # list of int scan numbers (folder.scan(int))

#folder.printscans()  # displays information on each scan

folder.print_hdf_address('entry1/scan_command')  # very fast display of hdf data

scan = folder(881430)
scan.plot.detail()

scans = folder.scans(all_scan_numbers, variables='scan_command')
print(scans)

# Create files showing data list
# scans.plot.plot_details_to_pdf(mydir + r"\OneDrive - Diamond Light Source Ltd\I16\Nexus_Format\example_nexus\all_scans.pdf")
# scans.plot.plot_details_to_html(mydir + r"\OneDrive - Diamond Light Source Ltd\I16\Nexus_Format\example_nexus\all_scans")
