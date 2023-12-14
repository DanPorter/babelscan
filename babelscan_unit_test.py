"""
Unit test for babelscan
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import babelscan


print('####################################################')
print('############## babelscan unit tests ################')
print('####################################################')
print('\n')
print(babelscan.module_info())

pth = os.path.expanduser('~')
file = pth + r"\Dropbox\Python\ExamplePeaks\810002.nxs"  # eta scan with pilatus
cv_file = pth + r"\Dropbox\Python\ExamplePeaks\857991.nxs"  # trajectory scan/ cvscan/ kthZebra
im_file = pth + r'\\OneDrive - Diamond Light Source Ltd\\I16\\Nexus_Format\\example_nexus\\872996.nxs'  # hkl scan with data
dat_file = pth + r'\\OneDrive - Diamond Light Source Ltd\\I16\\Nexus_Format\\example_nexus\\872996.dat'
datadir = pth + r"\OneDrive - Diamond Light Source Ltd\I16\Nexus_Format\example_nexus"  # eta scan with pilatus
rsmap = pth + r"\OneDrive - Diamond Light Source Ltd\I16\Nexus_Format\example_nexus\872996-pilatus3_100k-files\rsmap_872996_201215_101906.nxs"
i10_file = pth + r"\OneDrive - Diamond Light Source Ltd\I16\Nexus_Format\I10_nexus\i10-578596.nxs"
i06_file = pth + r"\OneDrive - Diamond Light Source Ltd\I16\Nexus_Format\I06_example\227980.dat"
i13_file = pth + r"\OneDrive - Diamond Light Source Ltd\I16\Nexus_Format\I13_example\i13-1-368910.nxs"
multid_file = pth + r"\OneDrive - Diamond Light Source Ltd\I16\Nexus_Format\example_nexus\928878.nxs"
pil2m = pth + r"\OneDrive - Diamond Light Source Ltd\I16\Nexus_Format\example_nexus\982681.nxs"


print('\n\n############ File Type Tests ##############')
print('standard I16 eta scan:')
scan = babelscan.file_loader(file)
print(scan)
print('\nI16 CV scan:')
scan = babelscan.file_loader(cv_file)
print(scan)
print('\nI16 hkl scan:')
scan = babelscan.file_loader(im_file)
print(scan)
print('\nI16 pil2m scan:')
scan = babelscan.file_loader(pil2m)
print(scan)
print('\nI16 .dat file:')
scan = babelscan.file_loader(dat_file)
print(scan)
print('\nI16 rsmap file:')
scan = babelscan.file_loader(rsmap)
print(scan)
print('\nI10 Nexus file:')
scan = babelscan.file_loader(i10_file)
print(scan)
print('\nI06 .dat file:')
scan = babelscan.file_loader(i06_file, scan_command_name='command')
print(scan)
print('\nI13 .Nexus file:')
scan = babelscan.file_loader(i13_file)
print(scan)


print('\n\n############ Missing count_time Tests ##############')
scan = babelscan.file_loader(file, debug='all')
scan.add2namespace(['count_time', 'counttime', 'Time', 't'], None, 'count_time')
print(scan)
print('\n\n')
print(scan('count_time'))
print('\n\n')
print(scan('nroi[31,31]'))


print('\n\n############### FolderMonitor Tests ################')
exp = babelscan.FolderMonitor(datadir)
scan = exp.scan(0)
print(scan)


print('\n\n##################### Plot Tests ###################')
scan = exp.scan(794940)
x, y, dy, xlab, ylab = scan.get_plot_data('axes', 'nroi_peak[31,31]', '/count_time/Transmission', 'np.sqrt(x+0.1)')

plt.figure()
plt.errorbar(x, y, dy, fmt='-o')
plt.xlabel(xlab)
plt.ylabel(ylab)
plt.title(scan.title())

scan.plot.image('sum', clim=[0, 100])
plt.show()

print('\n\n##################### Fit Tests ###################')
scan = exp(877619)  # merlin
scan.fit('axes', 'nroi_peak[31, 31]')
scan.plot('axes', ['nroi_peak[31, 31]', 'fit'])
print(scan.string('amplitude'))
print(scan.fit)

scan = exp.scan(794940)  # multipeak
scan.fit.multi_peak_fit(npeaks=2)
scan.plot('axes', ['signal', 'fit', 'p1_fit', 'p2_fit', 'bkg_fit'])
plt.show()

print('\n\n################# MultiScan Tests ##################')
scan_range = range(794932, 794947, 1)  # datadir, sperp, spara, eta scans
scans = exp.scans(scan_range, ['sperp', 'spara'])
print(scans)


print('\n\n################### Volume Tests ###################')
print('Image file: %s' % im_file)
scan = babelscan.file_loader(im_file)
volume = scan.volume()
print('%r, %s' % (scan, scan.find_image()))
print(volume)
print(np.max(volume))
print(volume.peak_search())

print('\n dat file: %s' % dat_file)
scan1 = babelscan.file_loader(dat_file)
volume1 = scan1.volume()
print('%r' % scan1)
print(volume1)
print(np.max(volume1))
print(volume1.peak_search())

print('\n Nexus file: %s' % file)
scan2 = babelscan.file_loader(file)
volume2 = scan2.volume()
print('%r, %s' % (scan2, scan2.find_image()))
print(volume2)
print(np.max(volume2))
print(volume2.peak_search())

print('\n Reciprocal space remapper file: %s' % rsmap)
scan3 = babelscan.file_loader(rsmap)
volume3 = scan3.volume()
print('%r, %s' % (scan3, scan3.find_image()))
print(volume3)
print(np.max(volume3))
print(volume3.peak_search())

print('\n Multi-dimensional scan file: %s' % multid_file)
scan4 = babelscan.file_loader(multid_file)
volume4 = scan4.volume()
print('%r, %s' % (scan4, scan4.find_image()))
print(volume3)
print(np.max(volume3))
print(volume3.peak_search())

# Volume plot
print('\n Test volume plotting')
volume2.plot()
am = np.array(volume2.argmax())
print('Volume argmax:', am, am - (10, 10, 10), am + (10, 10, 10))
from babelscan.plotting_matplotlib import create_axes, labels
ax = create_axes()
volume2.plot.cut(am-(10,10,10), am+(10,10,10), axes=ax)
labels('Volume', 'pixels', 'value', legend=True, axes=ax)
plt.show()


print('\n\n#################### Time Tests ####################')
allscan = exp.allscannumbers()
for scn in allscan:
    scan = exp.scan(scn)
    scan.options(start_time_name=['start_time', 'TimeSec'], end_time_name=['end_time', 'TimeSec'])
    scan.add2namespace(['counttime', 'Time', 't'], other_names='count_time', default_value=0)
    start_time = scan.time_start()
    duration = scan.duration()
    print('#%s  start: %s,  duration: %s' % (scn, start_time, duration))


print('\n\n#################### .dat Tests ####################')
exp.set_format('%d.dat')
allscan = exp.allscannumbers()
for scn in allscan:
    scan = exp.scan(scn)
    scan.options(start_time_name=['start_time', 'TimeSec'], end_time_name=['end_time', 'TimeSec'])
    scan.add2namespace(['counttime', 'Time', 't'], other_names='count_time', default_value=0)
    print(scn)
    start_time = scan.time_start()
    duration = scan.duration()
    print(scan)
    print('#%s  start: %s,  duration: %s' % (scn, start_time, duration))


print('\n\n########## More FolderMonitor Tests ################')
exp = babelscan.FolderMonitor(datadir)
# Add options
exp.options(
    str_list=['scan_number', 'scan_command', 'axes', 'signal', 'start_time', 'end_time', 'count_time'],
    start_time_name=['start_time', 'TimeSec'],
    end_time_name=['end_time', 'TimeSec'],
    names={'count_time': ['Time', 'counttime', 't']},
    defaults={'count_time': 0, 'start_time': None, 'end_time': None}
)
allfiles = exp.allscanfiles()
for f in allfiles:
    print(exp.scan(f))

print('\nAll tests completed. Hooray!')
