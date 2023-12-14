"""
BabelScan example script
Load a series of scan files, print the scan command and scan time
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import babelscan

mydir = os.path.expanduser('~')
instrument = babelscan.instrument_from_config('../config_files/i16.config')
exp = instrument.experiment(mydir + r"\OneDrive - Diamond Light Source Ltd\I16\Nexus_Format\example_nexus")

allscan = exp.allscanfiles()
for scn in allscan:
    try:
        scan = exp.scan(scn)
    except Exception:
        continue
    cmd = scan.string_format('{cmd}')
    scan.options(start_time_name=['start_time', 'TimeSec'], end_time_name=['end_time', 'TimeSec'])
    scan.add2namespace(['counttime', 'Time', 't'], other_names='count_time', default_value=0)
    start_time = scan.time_start()
    duration = scan.duration()
    tot_time = duration.seconds + duration.microseconds/1e6
    tot_points = scan.scan_length()
    print(f"    {{'cmd':'{cmd:60s}', 'time': {tot_time:8.2f}, 'points': {tot_points:4d}}}")

