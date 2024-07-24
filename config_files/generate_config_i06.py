"""
BabelScan
Create config file for i06
"""

import babelscan

# Names of datasets that will appear on print(scan)
str_list = ['scan_number', 'filename', 'scan_command', 'axes', 'signal',
            'i06_energy', 'i06_temperature']


# These will be added as **kwargs in each scan, see scan.options()
options = {
    'filename_format': 'i06-1-%06d.nxs',
    'label_command': '#{scan_number:1.0f}',
    'title_command': '{FolderTitle} #{scan_number:.0f} {i06_energy} {i06_temperature}\n{scan_command}',
    'scan_command_name': 'scan_command',
    'exposure_time_name': ['measurement/count_time', 'measurement/counttime', 'measurement/Time', 'measurment/t'],
    'start_time_name': ['start_time', 'TimeSec'],
    'end_time_name': ['end_time', 'TimeSec'],
    'axes_name': ['axes', 'xaxis'],
    'signal_name': ['signal', 'yaxis'],
    'str_list': str_list,
    'signal_operation': '/Transmission/count_time/(rc/300.)',
    'error_function': 'np.sqrt(x+0.1)',
    'instrument': 'i06'
}

# these will be added to alternative_names dict
default_names = {
    # name in namespace or hdf address: [shortcuts],
    'en': ['incident_energy', 'Energy', 'energy'],
    'Ta': ['Temperature', 'temp'],
    'Tb': ['Temperature', 'temp'],
    'Tgas': ['Temperature', 'temp'],
    'delta_axis_offset': ['do'],
    'counttime': ['count_time'],
    'Time': ['count_time'],
    't': ['count_time'],
}

# These will be added to the scan namespace using scan.string_format
default_formats = {
    'i06_temperature': '{Temperature:.3g}K',
    'i06_energy': '{incident_energy:.5g} eV',
}

# These will be added to the default_value dict and will be called if the name or alt_names can't be found
# this should contain all requested names in default_formats
default_values = {
    'FolderTitle': '',
    'scan_command': 'scan x y',
    'count_time': 1.0,
    'Transmission': 1.0,
    'en': 0.0,
    'Temperature': 300.0,
}

babelscan.save_to_config('i06.config', 'i06', default_names, default_formats, default_values, options)
