"""
Define Instrument class
 An instrument is a generator of Scans and FolderMonitors (experiements) with specific default settings
"""

from . import functions as fn
from . import file_loader, FolderMonitor, load_hdf_values


def instrument_from_config(config_file):
    """
    Create Instrument class from instrument.config file.
      .config files should be json files with the following keys:
        'name': str
        'default_names': dict,
        'formats': dict,
        'default_values': dict,
        'options': dict
    :param config_file: str config filename
    :return: Instrument
    """
    instr = Instrument('')
    instr.load_config_file(config_file)
    return instr


class Instrument:
    """
    Instrument class
     An instrument is a generator of Scans and FolderMonitors (experiements) with specific default settings

    beamline = Instrument('name', default_names, functions, filename_format)
    :param name: str : name of instrument
    :param default_names: dict : Scan objects created will
    :param formats: dict :
    :param options: dict :
    """
    def __init__(self, name, default_names=None, formats=None, default_values=None, options=None):
        self.name = name
        self._default_names = {} if default_names is None else default_names
        self._formats = {} if formats is None else formats
        self._default_values = {} if default_values is None else default_values
        self._options = {} if options is None else options
        self.filename = '%s.json' % name

    def __repr__(self):
        return "Instrument(%s)" % self.name

    def __str__(self):
        return '%r' % self

    def add_name(self, name, alt_names, default=None):
        """
        Set a name that will automatically be defined in the namespace
        :param name: str name that will appear in namespace
        :param alt_names: list of str alternative names that will return the same data as name
        :param default: any or None, if a search for name or alt_name returns no data, default will be used
        :return: None
        """
        self._default_names[name] = fn.liststr(alt_names)
        if default is not None:
            self._default_values[name] = default

    def add_format(self, name, operation):
        """
        Add a format operation
          add_format('i16_Energy', '{en:5.4f} keV')
        Note: Names referenced in format specifiers should have been added to the namespace using add_name
              otherwise errors may occur if the value can't be found
        :param name: str name of operation, will appear in scan namespace
        :param operation: str operation, calling namespace variables
        :return: None
        """
        self._formats[name] = operation

    def options(self, **kwargs):
        """Set or display options"""
        if len(kwargs) == 0:
            # return options
            out = 'Options:\n'
            for key, item in self._options.items():
                out += '%20s : %s\n' % (key, item)
            return out
        self._options.update(kwargs)

    def set_format(self, filename_format='%06d.nxs'):
        """Set the file format to monitor, uses printf-style string format, e.g. '%5d.nxs'"""
        self._options['filename_format'] = filename_format

    def set_str_list(self, names):
        """
        Set scan str_list - specifying which values to show on print(scan)
          set_str_list(['scan_command','axes','signal','en']
        :param names: list of str names
        :return: None
        """
        names = fn.liststr(names)
        self._options['str_list'] = names

    def set_error(self, error_op):
        """
        Set the default error operation
        :param error_op: function or str operation on 'x', e.g. 'np.sqrt(x+0.1)'
        :return: None
        """
        self._options['error_function'] = error_op

    def set_signal_operation(self, operation):
        """
        Set the operation to act on each can value when auto plotting (normalisation)
        :param operation: str operation, e.g. '/count_time/Transmission'
        :return: None
        """
        self._options['signal_operation'] = operation

    def _add_items(self, scan):
        """Add Insturment defaults to Scan"""
        scan.options(**self._options)
        for name, alt_names in self._default_names.items():
            scan.add2namespace(name, other_names=alt_names)
        for name, value in self._default_values.items():
            scan.add2namespace(name, default_value=value)
        for name, operation in self._formats.items():
            string = scan.string_format(operation)
            scan.add2namespace(name, string)

    def _scan_loader(self, filename, **kwargs):
        """Loads a babelscan.Scan and adds """
        scan = file_loader(filename, **kwargs)
        for name, alt_names in self._default_names.items():
            scan.add2namespace(name, other_names=alt_names)
        for name, value in self._default_values.items():
            scan.add2namespace(name, default_value=value)
        for name, operation in self._formats.items():
            string = scan.string_format(operation)
            scan.add2namespace(name, string)
        return scan

    def save_config_file(self, config_file=None):
        """
        Save config file
        :param config_file: str
        :return: None
        """
        if config_file is None:
            config_file = self.filename
        else:
            self.filename = config_file
        fn.save_to_config(config_file, self.name, self._default_names, self._formats,
                          self._default_values, self._options)

    def load_config_file(self, config_file=None):
        """
        Load values from config file
        :param config_file: str address of file
        :return: None
        """
        if config_file is None:
            config_file = self.filename
        name, default_names, formats, default_values, options = fn.load_from_config(config_file)
        self.name = name
        self._default_names.update(default_names)
        self._formats.update(formats)
        self._default_values.update(default_values)
        self._options.update(options)
        self.filename = config_file

    def experiment(self, directory, working_dir='.', **kwargs):
        """Create FolderMonitor"""
        options = self._options.copy()
        options.update(kwargs)
        return FolderMonitor(directory, working_dir, self._scan_loader, **options)

    def scan(self, filename, **kwargs):
        """return babelscan.Scan"""
        options = self._options.copy()
        data = {'FolderTitle': fn.file2foldername(filename)}
        if 'data' in options:
            options['data'].update(data)
        else:
            options['data'] = data
        options.update(kwargs)
        return self._scan_loader(filename, **options)

    def hdf_data(self, filenames, address, default=None):
        """Load HDF dataset data from multiple filenames"""
        return load_hdf_values(filenames, address, default)
