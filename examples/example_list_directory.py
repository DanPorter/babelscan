"""
BabelScan Example
List directory scan contents
"""

import babelscan

folder = babelscan.FolderMonitor(r"C:\Users\dgpor\OneDrive - Diamond Light Source Ltd\I16\Nexus_Format\example_nexus")

all_scan_files = folder.allscanfiles()  # list of str file names

all_scan_numbers = folder.allscannumbers()  # list of int scan numbers (folder.scan(int))

#folder.printscans()  # displays information on each scan

folder.print_hdf_address('entry1/scan_command')  # very fast display of hdf data

scan = folder(881430)
scan.plot.detail()

scans = folder.scans(all_scan_numbers, variables='scan_command')
#scans.plot.plot_details_to_pdf(r"C:\Users\dgpor\OneDrive - Diamond Light Source Ltd\I16\Nexus_Format\example_nexus\all_scans.pdf")

scans.plot.plot_details_to_html(r"C:\Users\dgpor\OneDrive - Diamond Light Source Ltd\I16\Nexus_Format\example_nexus\all_scans")


