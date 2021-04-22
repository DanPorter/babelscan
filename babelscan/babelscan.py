"""
babelscan object for holding many types of scan data
"""

import numpy as np
from . import functions as fn
from . import EVAL_MODE
from .settings import init_scan_plot_manager, init_multiscan_plot_manager, init_scan_fit_manager
from .volume import ArrayVolume


"----------------------------------------------------------------------------------------------------------------------"
"------------------------------------------------- Scan ---------------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"


class Scan:
    """
    Scan class
    Contains a namespace of data associated with names and a seperate dictionary of name associations,
    allowing multiple names to reference the same data.
      namespace = {
        'data1': [1,2,3],
        'data2': [10,20,30]
      }
      alt_names = {
        'xaxis': 'data1',
        'yaxis': 'data2',
      }
      dh = Scan(namespace, alt_names)
      dh('xaxis') >> returns [1,2,3]

    :param namespace: dict : dict of names and data {name: data}
    :param alt_names: dict or None* : dict of alternative names to names in namespace
    :param kwargs: key-word-argments as options shwon below, keywords and argmuents will be added to the namespace.

    Options:
      reload - True/False*, if True, reload mode is activated, reloading data on each operation
      label_name - str, add a name to use to automatically find the label
      label_command - str, format specifier for label, e.g. '{scan_number}'
      title_name - str, add a name to use to automatically find the title
      title_command - str, format specifier for title, e.g. '#{scan_number} Energy={en:5.2f} keV'
      scan_command_name - str, add a name to use to automatically find the scan command
      start_time_name - str, add a name to use to automatically find the start_time
      end_time_name - str, add a name to use to automatically find the end_time
      axes_name - str, add a name to use to automatically find the axes (xaxis)
      signal_name - str, add a name to use to automatically find the signal (yaxis)
      image_name - str, add a name to use to automatically find the detector image
      str_list - list of str, list of names to display when print(self)
      signal_operation - str, operation to perform on signal, e.g. '/Transmission'
      error_function - func., operation to perform on signal to generate errors. e.g. np.sqrt
      debug - str or list of str, options for debugging, options:
        'namespace' - displays when items are added to the namespace
        'eval' - displays when eval operations are used

    Functions
    add2namespace(name, data=None, other_names=None, hdf_address=None)
        set data in namespace
    add2strlist(names)
        Add to list of names in str output
    address(name)
        Return hdf address of namespace name
    array(names, array_length=None)
        Return numpy array of data with same length
    axes()
        Return default axes (xaxis) data
    dataset(name)
        Return dataset object
    eval(operation)
        Evaluate operation using names in dataset or in associated names
    find_image(multiple=False)
        Return address of image data in hdf file
    get_plot_data(xname=None, yname=None, signal_op=None, error_op=None)
        Return xdata, ydata, yerror, xname, yname
    image(idx=None, image_address=None)
        Load image from hdf file, works with either image addresses or stored arrays
    image_roi(cen_h=None, cen_v=None, wid_h=31, wid_v=31)
        Create new region of interest from detector images
    image_roi_op(operation)
        Create new region of interest (roi) from image data and return sum and maxval
    image_roi_sum(cen_h=None, cen_v=None, wid_h=31, wid_v=31)
        Create new region of interest
    image_size()
        Returns the image size
    label(new_label=None)
        Set or Return the scan label. The label is a short identifier for the scan, such as scan number
    load()
        Open and return hdf.File object
    name(name)
        Return corrected name from namespace
    options(**kwargs)
        Set or display options
    reload_mode(mode=None)
        Turns on reload mode - reloads the dataset each time
    reset()
        Reset the namespace
    scan_command()
        Returns scan command
    show_namespace()
        return str of namespace
    signal()
        Return default signal (yaxis) data
    string(names, str_format=None)
        Return formated string of data
    string_format(operation)
        Process a string with format specified in {} brackets, values will be returned.
    title(new_title=None)
        Set or Return the title
    tree(detail=False, recursion_limit=100)
        Return str of the full tree of data in a hdf object
    value(names, array_function=None)
        Return single value of data
    """
    def __init__(self, namespace, alt_names=None, default_values=None, **kwargs):
        self._namespace = kwargs.copy()
        self._alt_names = {}
        self._namespace.update(namespace)
        if alt_names is not None:
            self._alt_names.update(alt_names)
        self._default_values = {}
        if default_values is not None:
            self._default_values.update(default_values)

        # Managers
        self.plot = init_scan_plot_manager(self)
        self.fit = init_scan_fit_manager(self)

        # Options and defaults
        self._options = {}
        self._label_str = ['label']
        self._title_str = ['title', 'filename']
        self._scan_command_str = ['scan_command', 'cmd']
        self._start_time_str = ['start_time']
        self._end_time_str = ['end_time']
        self._exposure_time_str = ['count_time', 'counttime', 't']
        self._axes_str = ['axes', 'xaxis']
        self._signal_str = ['signal', 'yaxis']
        self._image_name = None
        self._image_size = None
        self._print_list = ['scan_command', 'axes', 'signal']
        self._reload_mode = False
        self._set_options(**kwargs)
        self._volume = None

        self._debug('init', '%r' % self)

    "------------------------------- Basic Operations -------------------------------------------"

    def reset(self):
        """Regenerate data lists"""
        self._namespace = {}

    def reload_mode(self, mode=None):
        """
        Turns on reload mode - reloads the dataset each time
        :param mode: Bool or None, True to turn on, None to return current mode
        :return: None or str
        """
        if mode is None:
            if self._reload_mode:
                return "Reload mode is ON"
            return "Reload mode is OFF"
        self._reload_mode = mode

    def add2namespace(self, name, data=None, other_names=None, default_value=None):
        """
        set data in namespace
        :param name: str name or list of names (each name will store the same data)
        :param data: any or None, data to store in namespace (nothing stored if None)
        :param other_names: str, list of str or None - strings to associate with name, giving the same result
        :param default_value: any or None, data to store in default_value namespace (nothing stored if None)
        :return: None
        """
        names = fn.liststr(name)
        if data is not None:
            for name in names:
                self._namespace[name] = data
                self._debug('namespace', 'Add to namespace: %s: %s' % (name, fn.data_string(data)))
        if other_names is not None:
            other_names = fn.liststr(other_names)
            for other_name in other_names:
                if other_name in self._alt_names:
                    self._alt_names[other_name] += names
                else:
                    self._alt_names[other_name] = names
                self._debug('namespace', 'Add alt. name: %s: %s' % (other_name, names))
        if default_value is not None:
            for name in names:
                self._default_values[name] = default_value
                self._debug('namespace', 'Add to namespace: %s: %s' % (name, fn.data_string(data)))

    def show_namespace(self):
        """return str of namespace"""
        out = 'Namespace %r:\n' % self
        out += '%-20s %-60s | %s\n' % ('Name', 'Alternate Names', 'Data')
        for key, item in self._namespace.items():
            other_names = ', '.join(okey for okey, oitem in self._alt_names.items() if key in oitem)
            out += '%-20s %-60s | %s\n' % (key, other_names, fn.data_string(item))
        return out

    def add2strlist(self, names):
        """Add to list of names in str output"""
        self._print_list += fn.liststr(names)

    def options(self, **kwargs):
        """Set or display options"""
        if len(kwargs) == 0:
            # return options
            out = 'Options:\n'
            for key, item in self._options.items():
                out += '%20s : %s\n' % (key, item)
            return out
        self._set_options(**kwargs)

    def _set_options(self, **kwargs):
        """Set options"""
        self._options.update(kwargs)
        if 'reload' in kwargs:
            self._reload_mode = kwargs['reload']
        if 'label_name' in kwargs:
            self._label_str = fn.liststr(kwargs['label_name'])
        if 'title_name' in kwargs:
            self._title_str = fn.liststr(kwargs['title_name'])
        if 'scan_command_name' in kwargs:
            self._scan_command_str = fn.liststr(kwargs['scan_command_name'])
        if 'start_time_name' in kwargs:
            self._start_time_str = fn.liststr(kwargs['start_time_name'])
        if 'end_time_name' in kwargs:
            self._end_time_str = fn.liststr(kwargs['end_time_name'])
        if 'exposure_time_name' in kwargs:
            self._exposure_time_str = fn.liststr(kwargs['exposure_time_name'])
        if 'axes_name' in kwargs:
            self._axes_str = fn.liststr(kwargs['axes_name'])
        if 'signal_name' in kwargs:
            self._signal_str = fn.liststr(kwargs['signal_name'])
        if 'image_name' in kwargs:
            self._image_name = fn.liststr(kwargs['image_name'])
        if 'str_list' in kwargs:
            self._print_list = fn.liststr(kwargs['str_list'])
        if 'scan_plot_manager' in kwargs:
            self.plot = kwargs['scan_plot_manager'](self)
        if 'scan_fit_manager' in kwargs:
            self.fit = kwargs['scan_fit_manager'](self)

    def _debug(self, debug_name, message):
        """
        Returns message if debug option active
        :param debug_name: str name to match in self._options['debug']
        :param message: str message to print if true
        :return: None
        """
        if 'debug' in self._options and debug_name in self._options['debug']:
            m = 'db:%s: %s' % (debug_name, message)
            print(m)
        elif 'debug' in self._options and 'all' in self._options['debug']:
            m = 'db:%s: %s' % (debug_name, message)
            print(m)

    "------------------------------- class operations -------------------------------------------"

    def __repr__(self):
        return 'Scan(namespace: %d, alt_names: %d)' % (len(self._namespace), len(self._alt_names))

    def __str__(self):
        out = self.__repr__()
        out += '\n' + '\n'.join(self.string(self._print_list))
        return out

    def __call__(self, name):
        return self.eval(name)

    def __getitem__(self, name):
        name, data = self._get_list_data(name)
        if len(data) == 1:
            return data[0]
        return data

    def __len__(self):
        return self.scan_length()

    def __add__(self, addee):
        """
        Add two scans together somehow
        """
        return MultiScan([self, addee])

    "------------------------------- data -------------------------------------------"

    def _load_data(self, name):
        """
        Check name in external dictionary, add to internal namespace
          This function will be overloaded in subclasses
        If 'name' not available, raise KeyError
        :param name: str
        """
        self._debug('load', 'Searching external databases for close match to: %s' % name)
        # Search for close match
        keys = [k.lower() for k in self._namespace.keys()]
        if name.lower() in keys:
            data = self._namespace[keys[keys.index(name.lower())]]
            self.add2namespace(name, data)
            return
        for key in keys:
            if name.lower() in key:
                data = self._namespace[key]
                self.add2namespace(name, data)
                return
        raise KeyError('\'%s\' not available in %r' % (name, self))

    def _get_name_data(self, name):
        """
        Get name and data from stored dicts
        Search hierachy:
          1. Check namespace for name, return namespace[name]
          2. Check alt_names for name, return namespace[alt_names[name][i]]
          3. Check name against special names in axes_str, signal_str
          4. Check name against 'nroi'
          5. Check name in external source e.g. hdf file
          6. Check name, alt_names in defaults_namespace
          7. If not available, raise KeyError
        :param name: str, key or associated key in namespace
        :return name, data: from namespace dict
        """
        if self._reload_mode:
            self._load_data(name)
        self._debug('load', 'Looking for %s in namespace, alt_names, specials' % name)
        if name in self._namespace:
            return name, self._namespace[name]
        if name in self._alt_names:
            for alt_name in self._alt_names[name]:
                if alt_name in self._namespace:
                    return alt_name, self._namespace[alt_name]
        # Check defaults
        if name in self._axes_str:  # e.g. 'axes'
            return self.axes()  # return _get_name_data('eta')
        if name in self._signal_str:
            return self.signal()
        # Check new region of interest
        if 'nroi' in name:
            roi_sum, roi_max = self.image_roi_op(name)
            return name, roi_sum
        # Load data from external dictionary (e.g. hdf file)
        # 'name' will be added to namespace, or KeyError will be raised
        # _load_data can be overloaded in subclasses
        try:
            self._load_data(name)
            return self._get_name_data(name)
        except KeyError as ke:
            # Finally, check the defaults namespace
            if name in self._default_values:
                self.add2namespace(name, self._default_values[name])
                return name, self._default_values[name]
            if name in self._alt_names:
                for alt_name in self._alt_names[name]:
                    if alt_name in self._default_values:
                        self.add2namespace(alt_name, self._default_values[name], name)
                        return alt_name, self._default_values[alt_name]
            raise ke

    def _get_data(self, name):
        return self._get_name_data(name)[1]

    def _get_list_data(self, names):
        """
        Get data from stored dicts
        :param names: str or list of str, key or associated key in namespace
        :return: list of data from namespace dict
        """
        names = fn.liststr(names)
        data = []
        new_name = []
        for name in names:
            n, d = self._get_name_data(name)
            data += [d]
            new_name += [n]
        return new_name, data

    def array(self, names, array_length=None):
        """
        Return numpy array of data with same length
         data with length 1 will be cast over the full length
         data with length >1 and < array_length will be filled with nans
        :param names: str or list of str, key or associated key in namespace
        :param array_length: int or None, length of arrays returned
        :return: array(n,array_length) where n is the length of list names
        """
        names, data = self._get_list_data(names)
        if array_length is None:
            array_length = np.max([np.size(d) for d in data])
        out = np.nan * np.zeros(shape=(len(data), array_length))
        for n in range(len(data)):
            if np.size(data[n]) == 1:
                out[n, :] = data[n]
            else:
                out[n, :len(data[n])] = data[n]
        return out

    def value(self, names, array_function=None):
        """
        Return single value of data
        :param names: str or list of str, key or associated key in namespace
        :param array_function: function to return a single value from an array
        :return: value or list of values
        """
        names, data = self._get_list_data(names)
        if array_function is None:
            array_function = fn.VALUE_FUNCTION
        out = [array_function(val) for val in data]
        if len(out) == 1:
            return out[0]
        return out

    def name(self, name):
        """
        Return corrected name from namespace
        :param name: str or list of str
        :return: str or list of str
        """
        names = np.asarray(name, dtype=str).reshape(-1)
        out = [self._get_name_data(name)[0] for name in names]
        if len(out) == 1:
            return out[0]
        return out

    def string(self, names, str_format=None):
        """
        Return formated string of data
        :param names: str or list of str, key or associated key in namespace
        :param str_format: format to use, e.g. '%s:%s'
        :return: str or list of str
        """
        names, data = self._get_list_data(names)
        if str_format is None:
            str_format = fn.OUTPUT_FORMAT
        out = [str_format % (name, fn.data_string(val)) for name, val in zip(names, data)]
        if len(out) == 1:
            return out[0]
        return out

    def time(self, names, date_format=None):
        """
        Return datetime object from data name
        :param names: str or list of str, key or associated key in namespace
        :param date_format: str format used in datetime.strptime (see https://strftime.org/)
        :return: list of datetime ojbjects
        """
        names, data = self._get_list_data(names)
        return fn.data_datetime(data, date_format)

    "------------------------------- Operations -----------------------------------------"

    def _prep_operation(self, operation):
        """
        prepare operation string, replace names with names in namespace
        :param operation: str
        :return operation: str, names replaced to match namespace
        """
        # First look for addresses in operation to seperate addresses from divide operations
        # addresses = fn.re_address.findall(operation)

        old_op = operation

        # Determine custom regions of interest 'nroi'
        rois = fn.re_nroi.findall(operation)
        for name in rois:
            new_name, data = self._get_name_data(name)
            operation = operation.replace(name, new_name)

        # Determine data for other variables
        names = fn.re_varname.findall(operation)
        for name in names:
            new_name, data = self._get_name_data(name)
            if new_name != name:
                operation = operation.replace(name, new_name)
        self._debug('eval', 'Prepare eval operation\n  initial: %s\n  final: %s' % (old_op, operation))
        return operation

    def _name_eval(self, operation):
        """
        Evaluate operation using names in dataset or in associated names
        :param operation: str
        :return: corrected operation, output of operation
        """
        if not EVAL_MODE:
            return self._get_name_data(operation)
        bad_names = ['import', 'os.', 'sys.']
        for bad in bad_names:
            if bad in operation:
                raise Exception('This operation is not allowed as it contains: "%s"' % bad)
        operation = self._prep_operation(operation)
        result = eval(operation, globals(), self._namespace)
        if operation in self._namespace or operation in self._alt_names:
            return operation, result
        # add to namespace
        n = 1
        while 'operation%d' % n in self._namespace:
            n += 1
        self.add2namespace('operation%d' % n, result, operation)
        return operation, result

    def eval(self, operation):
        """
        Evaluate operation using names in dataset or in associated names
        :param operation: str
        :return: output of operation
        """
        _, out = self._name_eval(operation)
        return out

    def string_format(self, operation):
        """
        Process a string with format specified in {} brackets, values will be returned.
        e.g.
          operation = 'the energy is {energy} keV'
          out = string_command(operation)
          # energy is found within hdf tree
          out = 'the energy is 3.00 keV'
        :param operation: str format operation e.g. '#{scan_number}: {title}'
        :return: str
        """
        # get values inside brackets
        ops = fn.re_strop.findall(operation)
        format_namespace = {}
        for op in ops:
            op = op.split(':')[0]  # remove format specifier
            name, data = self._name_eval(op)
            try:
                value = fn.VALUE_FUNCTION(data)
            except TypeError:
                value = data
            format_namespace[name] = value
            operation = operation.replace(op, name)
        return operation.format(**format_namespace)

    def _get_error(self, name, operation=None):
        """
        Return uncertainty on data using operation
        :param operation: function to apply to signal, e.g. 'np.sqrt'
        :param operation: None* will default to zero, unless "error_function" in options
        :return: operation(array)
        """
        _, data = self._name_eval(name)
        if operation is None:
            if 'error_function' in self._options:
                operation = self._options['error_function']
            else:
                return np.zeros(np.shape(data))
        return operation(data)

    def _get_signal_operation(self, name, signal_op=None, error_op=None):
        """
        Return data after operation with error
        :param name: str name in namespace
        :param signal_op: operation to perform on name, e.g. '/Transmission'
        :param error_op: function to performon name, e.g. np.sqrt
        :return: signal_name, output, error arrays
        """
        name, data = self._name_eval(name)
        error = self._get_error(name, error_op)
        # add error array to namespace
        error_name = '%s_error' % name
        self.add2namespace(error_name, error)

        if signal_op is None:
            if 'signal_operation' in self._options:
                signal_op = self._options['signal_operation']
            else:
                return name, data, error
        # Create operations
        operation = name + signal_op
        operation_error = error_name + signal_op
        signal = self.eval(operation)
        error = self.eval(operation_error)
        return operation, signal, error

    "------------------------------- Defaults -------------------------------------------"

    def label(self, new_label=None):
        """
        Set or Return the scan label. The label is a short identifier for the scan, such as scan number
        :param new_label: str or None, if str sets the label as str, if None, returns automatic label
        :return: None or str
        """
        if new_label:
            self.add2namespace(self._label_str[0], new_label)
            return

        if 'label_command' in self._options:
            return self.string_format(self._options['label_command'])

        add2othernames = []
        for s in self._label_str:
            try:
                data = self._get_data(s)
                self.add2namespace(s, other_names=add2othernames)
                return data
            except KeyError:
                add2othernames += [s]
        raise Exception('No label in %r' % self)

    def title(self, new_title=None):
        """
        Set or Return the title
        :param new_title: str or None, if str sets the title as str, if None, returns automatic title
        :return: None or str
        """
        if new_title:
            self.add2namespace(self._title_str[0], new_title)
            return

        if 'title_command' in self._options:
            return self.string_format(self._options['title_command'])

        add2othernames = []
        for s in self._title_str:
            try:
                data = self._get_data(s)
                self.add2namespace(s, other_names=add2othernames)
                return data
            except KeyError:
                add2othernames += [s]
        raise Exception('No title in %r' % self)

    def scan_command(self):
        """
        Returns scan command
        :return: str
        """
        add2othernames = []
        for s in self._scan_command_str:
            try:
                data = self._get_data(s)
                self.add2namespace(s, other_names=add2othernames)
                return data
            except KeyError:
                add2othernames += [s]
        raise Exception('No Scan Command in %r' % self)

    def time_start(self):
        """
        Return scan time_start time
        :return: datetime
        """
        add2othernames = []
        for s in self._start_time_str:
            try:
                data = self.time(s)[0]
                self.add2namespace(s, other_names=add2othernames)
                return data
            except KeyError:
                add2othernames += [s]
        raise Exception('No time_start time in %r' % self)

    def time_end(self):
        """
        Return scan end time
        :return: datetime
        """
        add2othernames = []
        for s in self._end_time_str:
            try:
                data = self.time(s)[-1]
                self.add2namespace(s, other_names=add2othernames)
                return data
            except KeyError:
                add2othernames += [s]
        for s in self._start_time_str[::-1]:
            try:
                data = self.time(s)[-1]
                self.add2namespace(s, other_names=add2othernames)
                return data
            except KeyError:
                add2othernames += [s]
        raise Exception('No end time in %r' % self)

    def duration(self, start_time=None, end_time=None):
        """
        Calculate time difference between two times
        :param start_time: str name of date dataset or array of timestamps
        :param end_time: None or str name of date dataset
        :return: datetime.timedelta
        """
        if end_time is not None:
            end_time = self.time(end_time)[-1]
        if start_time is None:
            start_time = self.time_start()
        else:
            lst = self.time(start_time)
            start_time = lst[0]
            if len(lst) > 1 and end_time is None:
                end_time = lst[-1]
        if end_time is None:
            end_time = self.time_end()
        return end_time - start_time

    def exposure_time(self):
        """Return the exposure time"""
        value = 1.0
        for s in self._exposure_time_str:
            try:
                value = self.value(s)
            except KeyError:
                pass
        self.add2namespace('exposure_time', value, other_names=self._exposure_time_str)
        return value

    def _find_defaults(self):
        """
        Find default axes and signal (x-axis/y-axis), adds to namespace
         This function may be overloaded in subclasses
        :return: axes_name, signal_name
        """
        scan_command = self.scan_command()
        # axes / x-axis
        axes_name = fn.axes_from_cmd(scan_command)
        axes_data = self._get_data(axes_name)
        self.add2namespace(axes_name, axes_data, self._axes_str)
        # signal / y-axis
        signal_name = fn.signal_from_cmd(scan_command)
        signal_data = self._get_data(signal_name)
        self.add2namespace(signal_name, signal_data, self._signal_str)
        return axes_name, signal_name

    def axes(self):
        """
        Return default axes (xaxis) data
        :return: array
        """
        add2othernames = []
        for name in self._axes_str:
            if name in self._namespace:
                self.add2namespace(name, other_names=add2othernames)
                return self._namespace[name]
            if name in self._alt_names:
                for alt_name in self._alt_names[name]:
                    if alt_name in self._namespace:
                        self.add2namespace(name, other_names=add2othernames)
                        return self._namespace[alt_name]
            add2othernames += [name]
        # axes not in namespace, get from scan command
        axes_name, signal_name = self._find_defaults()
        return self._get_name_data(axes_name)

    def signal(self):
        """
        Return default signal (yaxis) data
        :return: array
        """
        add2othernames = []
        for name in self._signal_str:
            if name in self._namespace:
                self.add2namespace(name, other_names=add2othernames)
                return self._namespace[name]
            if name in self._alt_names:
                for alt_name in self._alt_names[name]:
                    if alt_name in self._namespace:
                        self.add2namespace(name, other_names=add2othernames)
                        return self._namespace[alt_name]
            add2othernames += [name]
        # signal not in namespace, get from scan command
        axes_name, signal_name = self._find_defaults()
        return self._get_name_data(signal_name)

    def scan_length(self):
        """
        Return the number of points in the scan (length of 'axes')
        :return: int
        """
        return np.size(self.axes())

    def get_plot_data(self, xname=None, yname=None, signal_op=None, error_op=None):
        """
        Return xdata, ydata, yerror, xname, yname
         x, y, dy, xlabel, ylabel = scan.get_plot_data('axes', 'signal', '/Transmission', np.sqrt)

        :param xname: str name of value to use as x-axis
        :param yname: str name of value to use as y-axis
        :param signal_op: operation to perform on yaxis, e.g. '/Transmission'
        :param error_op: function to use on yaxis to generate error, e.g. np.sqrt
        :return xdata: array
        :return ydata: array
        :return yerror: array
        :return xname: str
        :return yname: str
        """
        if xname is None:
            xname = self._axes_str[0]
        if yname is None:
            yname = self._signal_str[0]
        xname, xdata = self._get_name_data(xname)
        yname, ydata, yerror = self._get_signal_operation(yname, signal_op, error_op)
        return xdata, ydata, yerror, xname, yname

    def save_plot_data(self, filename=None, xname=None, yname=None, signal_op=None, error_op=None):
        """
        Return xdata, ydata, yerror, xname, yname
         x, y, dy, xlabel, ylabel = scan.get_plot_data('axes', 'signal', '/Transmission', np.sqrt)

        :param filename: str filename to save data to
        :param xname: str name of value to use as x-axis
        :param yname: str name of value to use as y-axis
        :param signal_op: operation to perform on yaxis, e.g. '/Transmission'
        :param error_op: function to use on yaxis to generate error, e.g. np.sqrt
        :return xdata: array
        :return ydata: array
        :return yerror: array
        :return xname: str
        :return yname: str
        """
        xdata, ydata, yerror, xlabel, ylabel = self.get_plot_data(xname, yname, signal_op, error_op)

        if filename is None:
            filename = 'Scan_%s_%s_%s.csv' % (self.label(), xname, yname)

        with open(filename, 'wt') as f:
            f.write('# %r\n' % self)
            title = '\n# '.join(self.title().split('\n'))
            f.write('# %s\n' % title)
            labels = ','.join([xlabel, ylabel, 'error'])
            f.write('# %s\n' % labels)
            for x, y, e in zip(xdata, ydata, yerror):
                f.write('%s, %s, %s\n' % (x, y, e))
        print('(%s, %s, error) written to %s' % (xlabel, ylabel, filename))


    "------------------------------- images -------------------------------------------"

    def image(self, idx):
        """
        Return detector image
         Overloaded in subclasses, this version does nothing interesting
        :param idx: int index of image
        :return: 2d array
        """
        return np.zeros([100, 100])

    def volume(self):
        """
        Rerturn volume
        :return:
        """
        if self._volume is None:
            vol = np.array([self.image(idx) for idx in range(self.scan_length())])
            self._volume = ArrayVolume(vol)
        return self._volume

    def image_size(self):
        """
        Returns the image size
        :return: tuple
        """
        if self._image_size:
            return self._image_size
        image = self.image(0)
        shape = np.shape(image)
        self._image_size = shape
        return shape

    def image_roi(self, cen_h=None, cen_v=None, wid_h=31, wid_v=31):
        """
        Create new region of interest from detector images
        :param cen_h: int or None
        :param cen_v: int or None
        :param wid_h:  int or None
        :param wid_v:  int or None
        :return: l*v*h array
        """
        return self.volume().roi(cen_h, cen_v, wid_h, wid_v)

    def image_roi_sum(self, cen_h=None, cen_v=None, wid_h=31, wid_v=31):
        """
        Create new region of interest
        :param cen_h: int or None
        :param cen_v: int or None
        :param wid_h:  int or None
        :param wid_v:  int or None
        :return: roi_sum, roi_max
        """
        volume = self.volume()
        roi_sum, roi_max = volume.roi_sum(cen_h, cen_v, wid_h, wid_v)
        # Add to namespace
        n = 1
        while 'nroi%d_sum' % n in self._namespace:
            n += 1
        full_name = 'nroi[%d,%d,%d,%d]' % (cen_h, cen_v, wid_h, wid_v)
        self.add2namespace('nroi%d_sum' % n, roi_sum, other_names=full_name)
        self.add2namespace('nroi%d_max' % n, roi_max)
        return roi_sum, roi_max

    def image_roi_op(self, operation):
        """
        Create new region of interest (roi) from image data and return sum and maxval
        The roi centre and size is defined by an operation:
          operation = 'nroi[210, 97, 75, 61]'
          'nroi'      -   creates a region of interest in the detector centre with size 31x31
          'nroi[h,v]' -   creates a roi in the detector centre with size hxv, where h is horizontal, v is vertical
          'nroi[m,n,h,v] - create a roi with cen_h, cen_v, wid_h, wid_v = n, m, h, v
        :param operation: str : operation string
        :return: sum, maxval : [o] length arrays
        """
        volume = self.volume()
        cen_h, cen_v, wid_h, wid_v, _ = volume.check_roi_op(operation)
        roi_sum, roi_max = self.image_roi_sum(cen_h, cen_v, wid_h, wid_v)
        # Add operation to associated namespace
        n = 1
        while 'nroi%d_sum' % n in self._namespace:
            n += 1
        n -= 1
        name = 'nroi%d_sum' % n
        self._debug('nroi', 'New ROI created: %s, saved in namespace as %s' % (operation, name))
        self.add2namespace(name, other_names=operation)
        return roi_sum, roi_max


"----------------------------------------------------------------------------------------------------------------------"
"----------------------------------------------- MultiScan ------------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"


class MultiScan:
    """
    Class for holding multiple DataHolders
    """
    def __init__(self, scan_list, variables=None):
        self._scan_list = []
        for scan in scan_list:
            if issubclass(type(scan), MultiScan):
                self._scan_list.extend(scan._scan_list)
            else:
                self._scan_list.append(scan)

        if variables is None:
            self._variables = []
        else:
            self._variables = list(np.asarray(variables, dtype=str).reshape(-1))

        # Managers
        self.plot = init_multiscan_plot_manager(self)

    def __repr__(self):
        return 'MultiScan(%d items)' % len(self._scan_list)

    def __str__(self):
        variables = self.string(self._variables)
        out = ''
        for n in range(len(self._scan_list)):
            out += '%3d %s: %s\n' % (n, self._scan_list[n].label(), variables[n])
        return out

    def __add__(self, other):
        return MultiScan([self, other])

    def __call__(self, name):
        return [scan(name) for scan in self._scan_list]

    def __getitem__(self, item):
        return self._scan_list[item]

    def __len__(self):
        return len(self._scan_list)

    def add_variable(self, name):
        """
        Add variable
        :param name: name of variable parameter between scans
        :return:
        """
        self._variables.append(name)

    def _get_variable_data(self):
        """
        Return array of variable data such that
          data = self._get_variable_data()
          data[0] == data for self._variables[0]
        """
        return np.transpose(self.value(self._variables)).reshape(len(self._variables), -1)

    def _get_variable_string(self):
        """
        Return string of variable data
        :return: str
        """
        return '\n'.join(self.string(self._variables))

    def _get_name(self, name):
        """
        Return corrected name from first scan
        :param name: str
        :return: str
        """
        try:
            name = self._scan_list[0].name(name)
        except (IndexError, KeyError):
            pass
        return name

    def array(self, name, array_length=None):
        data = self.__call__(name)
        if array_length is None:
            array_length = np.max([np.size(d) for d in data])
        return np.array([scan.array(name, array_length)[0] for scan in self._scan_list])

    def value(self, name):
        return [scan.value(name) for scan in self._scan_list]

    def string(self, name):
        out = []
        for scan in self._scan_list:
            strlist = np.asarray(scan.string(name), dtype=str).reshape(-1)
            out += [', '.join(s.strip() for s in strlist)]
        return out

    def string_format(self, command):
        return [scan.string_format(command) for scan in self._scan_list]

    def griddata(self, axes=None, signal='signal', repeat_after=None):
        """
        Generate 2D square grid of single values for each scan
        Return x, y axis when taking single values from each scan
        :param axes: str or list of str, names of axes data
        :param signal: str name of signal data
        :param repeat_after: int or None, defines repeat length of data
        :return: xaxis, yaxis, zaxis square [n,m] arrays
        """
        if axes is None:
            axes = self._variables
        else:
            axes = np.asarray(axes).reshape(-1)

        if len(axes) > 0:
            xaxis = self.value(axes[0])
        else:
            xaxis = np.arange(len(self._scan_list))

        yaxis = self.value(axes[-1])
        zaxis = self.value(signal)
        xaxis, yaxis, zaxis = fn.square_array(xaxis, yaxis, zaxis, repeat_after)
        return xaxis, yaxis, zaxis

    def get_plot_variable(self, yname, variable=None):
        """
        Get plotting data for plotting data points from each scan
         x, y, xlabel, ylabel = scans.get_plot_variable('signal', 'scanno')
        e.g.
         for n in range(len(x)):
            plt.plot(x[n], y[n])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        :param yname: str name of value to use as y-axis
        :param variable: str, list of str or None for default list of scans variables
        :return xdata: list of arrays for each scan
        :return ydata: list of arrays for each scan
        :return yerror: list of arrays for each scan
        :return labels: list of str for each scan
        :return xlabel: str, axis label for x-axis
        :return ylabel: str, axis label for y-axis
        """
        if variable is None:
            variable = []
        variables = list(np.asarray(variable, dtype=str).reshape(-1)) + self._variables
        xlabel = variables[0]
        xdata = self.value(xlabel)
        ylabel = yname
        ydata = self.value(yname)
        return xdata, ydata, xlabel, ylabel

    def get_plot_lines(self, xname=None, yname=None, signal_op=None, error_op=None):
        """
        Get plotting data for plotting as series of lines
         x, y, dy, labels, xlabel, ylabel = scans.get_plot_lines('axes', 'signal', '/Transmission', np.sqrt)
        e.g.
         for n in range(len(x)):
            plt.errorbar(x[n], y[n], dy[n])
        plt.legend(labels)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        :param xname: str name of value to use as x-axis
        :param yname: str name of value to use as y-axis
        :param signal_op: operation to perform on yaxis, e.g. '/Transmission'
        :param error_op: function to use on yaxis to generate error, e.g. np.sqrt
        :return xdata: list of arrays for each scan
        :return ydata: list of arrays for each scan
        :return yerror: list of arrays for each scan
        :return labels: list of str for each scan
        :return xlabel: str, axis label for x-axis
        :return ylabel: str, axis label for y-axis
        """
        xdata = []
        ydata = []
        dydata = []
        xlabel, ylabel = xname, yname
        for scan in self._scan_list:
            x, y, dy, xlabel, ylabel = scan.get_plot_data(xname, yname, signal_op, error_op)
            xdata += [x]
            ydata += [y]
            dydata += [dy]
        labels = self.__str__().splitlines()
        return xdata, ydata, dydata, labels, xlabel, ylabel

    def get_plot_mesh(self, xname=None, yname=None, signal_op=None, error_op=None):
        """
        Return array data for plotting as mesh
         x, y, z, xlabel, ylabel, zlabel = scans.get_plot_mesh('axes', 'signal', '/Transmission', np.sqrt)
        e.g.
         plt.pcolormesh(x, y, z)
         plt.xlabel(xlabel)
         plt.ylabel(ylabel)

        :param xname: str name of value to use as x-axis
        :param yname: str name of value to use as y-axis
        :param signal_op: operation to perform on yaxis, e.g. '/Transmission'
        :param error_op: function to use on yaxis to generate error, e.g. np.sqrt
        :return xdata: list of arrays for each scan
        :return ydata: list of arrays for each scan
        :return yerror: list of arrays for each scan
        :return labels: list of str for each scan
        :return xlabel: str, axis label for x-axis
        :return ylabel: str, axis label for y-axis
        """
        xname = self._get_name(xname)
        yname = self._get_name(yname)
        pass


