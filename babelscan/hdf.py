"""
Subclass data holder for .hdf and .nxs files
"""

import os
import re
import datetime
import numpy as np
import h5py
try:
    import hdf5plugin  # required for compressed data
except ImportError:
    print('Warning: hdf5plugin not available.')

from . import functions as fn
from .babelscan import Scan
from .volume import ImageVolume, DatasetVolume, ArrayVolume


"----------------------------LOAD FUNCTIONS---------------------------------"


def load(filename):
    """Load a hdf5 or nexus file"""
    try:
        return h5py.File(filename, 'r')
    except OSError:
        if os.path.isfile(filename):
            raise Exception('File not readable, maybe you need to import hdf5plugin')
        raise Exception('File does not exist or is currently being written: %s' % filename)


def reload(hdf):
    """Reload a hdf file, hdf = reload(hdf)"""
    filename = hdf.filename
    return load(filename)


def load_hdf_values(files, address, default=None):
    """
    Load single dataset value (metadata) from hdf files
      Will return str or float value as per dataset. Array datsets will be averaged to return a single float.
    :param files: str or list of str file names
    :param address: str hdf dataset address
    :param default: value to return if dataset not in file
    :return: array of floats or strings
    """
    files = fn.liststr(files)
    values = np.empty(len(files), dtype=object)
    for n, file in enumerate(files):
        with load(file) as hdf:
            if address in hdf:
                dataset = hdf.get(address)
                if dataset.ndim > 0:
                    values[n] = np.mean(dataset)
                else:
                    values[n] = hdf.get(address)[()]
            else:
                values[n] = default
    return values


"--------------------------DATASET FUNCTIONS--------------------------------"


def address_name(address):
    """Convert hdf address to name"""
    return os.path.basename(address)


def address_group(address, group_name=None):
    """
    Return part of address upto group_name
    :param address: str hdf address
    :param group_name: str name of group
    :return: reduced str
    """
    if group_name is None:
        names = address.replace('\\', '/').split('/')
        return '/'.join(names[:-1])
    return re.findall(r'(.+?%s.*?)(?:\/|$)' % group_name, address, re.IGNORECASE)[0]


def address_group_name(address):
    """
    Return name of dataset group /entry/[group]/name
    :param address: str hdf address
    :return: str
    """
    names = address.replace('\\', '/').split('/')
    return names[-2]


def is_dataset(dataset):
    """
    Check if input is a hdf dataset
     e.g. is_dataset(hdf_group.get(address))
    """
    return hasattr(dataset, 'size')


def is_group(dataset):
    """
    Check if input is a hdf group
    :param dataset:
    :return: True/ False
    """
    return hasattr(dataset, 'keys')


def dataset_name(dataset):
    """
    Return name of the dataset
    the name is the final part of the hdf dataset address
    equivalent to:
      dataset_name = dataset.name.split('/')[-1]
    Warning - dataset.name is not always stored as the correct value
    """
    return address_name(dataset.name)


def dataset_data(dataset):
    """Get data from dataset, return float, array or str"""
    # convert arrays of length 1 to values
    if not dataset:
        return None
    if dataset.size == 1 and len(dataset.shape) == 1:
        data = np.asarray(dataset)[0]
    else:
        data = dataset[()]
    # Handle bytes strings to return string
    try:
        data = data.decode(fn.BYTES_DECODER)
    except (UnicodeDecodeError, AttributeError):
        pass
    return data


def dataset_string(dataset):
    """Generate string from dataset"""
    data = dataset_data(dataset)
    try:
        # single value
        return fn.VALUE_FORMAT % data
    except TypeError:
        # array
        if dataset.size > 1:
            return fn.data_string(data)
    # probably a string
    return fn.shortstr('%s' % data)


def dataset_datetime(dataset, input_format=None, output_format=None):
    """
    Read time stamps from hdf file at specific address
    If input is a string (or bytes), input_format is used to parse the string
    If input is a float, it is assumed to be a timestamp from the Unix Epoch (1970-01-01 00:00:00)

    Useful Format Specifiers (https://strftime.org/):
    %Y year         %m month      %d day      %H hours    %M minutes  %S seconds  %f microseconds
    %y year (short) %b month name %a day name %I 12-hour  %p AM or PM %z UTC offset

    :param dataset: hdf dataset
    :param input_format: str datetime.strptime format specifier to parse dataset
    :param output_format: str datetime.strftime format specifier to generate output string (if None, returns datetime)
    :return datetime or list of datetime
    """
    if input_format is None:
        input_format = fn.DATE_FORMAT
    data = dataset_data(dataset)
    data = np.asarray(data, dtype=str).reshape(-1)
    try:
        # str date passed, e.g. start_time: '2020-10-22T09:33:11.894+01:00'
        dates = np.array([datetime.datetime.strptime(date, input_format) for date in data])
    except ValueError:
        # float timestamp passed, e.g. TimeFromEpoch: 1603355594.96
        dates = np.array([datetime.datetime.fromtimestamp(float(time)) for time in data])

    if output_format:
        if len(data) == 1:
            return dates[0].strftime(output_format)
        else:
            return [date.strftime(output_format) for date in dates]
    else:
        if len(data) == 1:
            return dates[0]
        return dates


def show_attrs(dataset):
    """Return formatted string of attributes for hdf object"""
    out = '%s with %d attrs\n' % (dataset, len(dataset.attrs))
    out += '%s\n' % dataset.name
    for key, value in dataset.attrs.items():
        out += '%30s : %s\n' % (key, value)
    return out


def get_attribute(dataset, attribute='NX_class', default=''):
    """Return a specific attribute of dataset or group"""
    if attribute in dataset.attrs:
        return np.squeeze(dataset.attrs[attribute]).astype(str)
    return default


"-------------------------HDF ADDRESS FUNCTIONS-------------------------------"


def dataset_addresses(hdf_group, addresses='/', recursion_limit=100, get_size=None, get_ndim=None):
    """
    Return list of addresses of datasets, starting at each address
    :param hdf_group: hdf5 File or Group object
    :param addresses: list of str or str : time_start in this / these addresses
    :param recursion_limit: Limit on recursivley checking lower groups
    :param get_size: None or int, if int, return only datasets with matching size
    :param get_ndim: None or int, if int, return only datasets with matching ndim
    :return: list of str
    """
    addresses = np.asarray(addresses, dtype=str).reshape(-1)
    out = []
    for address in addresses:
        data = hdf_group.get(address)
        if data and is_dataset(data):
            # address is dataset
            if (get_size is None and get_ndim is None) or (get_size is not None and data.size == get_size) or (
                    get_ndim is not None and data.ndim == get_ndim):
                out += [address]
        elif data and recursion_limit > 0:
            # address is Group
            new_addresses = ['/'.join([address, d]).replace('//', '/') for d in data.keys()]
            out += dataset_addresses(hdf_group, new_addresses, recursion_limit - 1, get_size, get_ndim)
        #elif recursion_limit > 0:
        #    # address is None, search for group address and iterate
        #    new_address = get_address(hdf_group, address, return_group=True)  # this goes forever if a group fails to load
        #    if new_address:
        #        out += dataset_addresses(hdf_group, new_address, recursion_limit - 1, get_size, get_ndim)
    return out


def find_name(name, address_list, match_case=False, whole_word=False):
    """
    Find datasets using field name
    :param name: str : name to match in dataset field name
    :param address_list: list of str: list of str to search in
    :param match_case: if True, match case of name
    :param whole_word: if True, only return whole word matches
    :return: list of str matching dataset addresses
    """
    out = []
    if not match_case: name = name.lower()
    for address in address_list:
        a_name = (address_name(address) if whole_word else address)
        a_name = (a_name if match_case else a_name.lower())
        if whole_word and name == a_name:
            out += [address]
        elif not whole_word and name in a_name:
            out += [address]
    return out


def find_cascade(name, address_list, exact_only=False, find_any=False):
    """
    Find dataset using field name in a cascading fashion:
        1. Find exact match (matching case, whole_word)
        2. any case, whole_word
        3. any case, anywhere in address
        4. Return None otherwise
    :param name: str : name to match in dataset field name
    :param address_list: list of str: list of str to search in
    :param exact_only: return list of exact matches only (may be length 0)
    :param find_any: if True, return matches where string appears anywhere in address
    :return: list of str addresses matching name
    """
    # fast return of full address
    if address_list.count(name) == 1:
        return [name]

    if '/' in name:
        # address, or part of address given.
        # Addresses are unique but exact match not found, return closest match
        return [address for address in address_list if address.lower().endswith(name.lower())]

    # only match the address name
    name_list = [address_name(address) for address in address_list]

    # Exact match
    exact_match = [address for idx, address in enumerate(address_list) if name == name_list[idx]]
    if exact_match or exact_only:
        return exact_match

    # If not found, try matching lower case
    lower_match = [address for idx, address in enumerate(address_list) if name.lower() == name_list[idx].lower()]
    if lower_match:
        return lower_match

    # If not found, try matching any
    if find_any:
        any_match = [address for address in address_list if address.lower().endswith(name.lower())]
        if any_match:
            return any_match

    # If not found, try matching group
    group_match = [address for address in address_list if name == address_group_name(address)]
    return group_match


def tree(hdf_group, detail=False, groups=False, recursion_limit=100):
    """
    Return str of the full tree of data in a hdf object
    :param hdf_group: hdf5 File or Group object
    :param detail: False/ True - provide further information about each group and dataset
    :param groups: False/ True - only display group level structure
    :param recursion_limit: int max number of levels
    :return: str
    """
    if recursion_limit < 1: return ''
    outstr = '%s\n' % hdf_group.name
    if detail:
        for attr, val in hdf_group.attrs.items():
            outstr += '  @%s: %s\n' % (attr, val)
    try:
        for branch in hdf_group.keys():
            new_group = hdf_group.get(branch)
            if new_group:
                outstr += tree(new_group, detail, groups, recursion_limit-1)
        return outstr
    except AttributeError:
        # doesn't have .keys(), hdf_group = dataset, should have .name, .size, .shape
        if groups:
            out = ""
        elif detail:
            try:
                ds_string = dataset_string(hdf_group)
            except OSError:
                # Catch missing datasets
                ds_string = 'Not Available! (%s)' % (hdf_group.file)
            out = '  %s: %s\n' % (hdf_group.name, ds_string)
            for attr, val in hdf_group.attrs.items():
                out += '    @%s: %s\n' % (attr, val)
        else:
            out = '  %s, size: %s, shape: %s\n' % (hdf_group.name, hdf_group.size, hdf_group.shape)
        return out


def tree_debug(hdf_group):
    """
    print full tree structrue of hdf including attributes
    :param hdf_group: hdf5 File or Group object
    :return: None
    """
    try:
        keys = hdf_group.keys()
        print('\nHDF Group: ', hdf_group)
        print('  Name: ', hdf_group.name)
        for attr, val in hdf_group.attrs.items():
            print('  Attr: ', attr, val)
        print('  Keys: ', list(keys))
        for key in keys:
            tree_debug(hdf_group.get(key))

    except AttributeError:
        print('  Dataset: ', hdf_group)
        print('    Name: ', hdf_group.name)
        for attr, val in hdf_group.attrs.items():
            print('    Attr: ', attr, val)


def nexus_tree(hdf_group, list_datasets=False):
    """
    Return str of nexus structure of HDF file
    :param hdf_group: hdf5 File or Group object
    :param list_datasets: if True, list class of each dataset
    :return: str
    """
    top_level_groups = tree(hdf_group, groups=True, recursion_limit=3).splitlines()
    out = 'File: %s\n' % hdf_group.filename
    out += 'Groups:\n'
    for group in top_level_groups:
        datasets = dataset_addresses(hdf_group, group, recursion_limit=1)
        if len(datasets) < 1: continue
        names = [address_name(address) for address in datasets]
        nxclass = [get_attribute(hdf_group[address], 'NX_class', 'HDFdataset') for address in datasets]
        out += ' %s:  %s\n   ' % (get_attribute(hdf_group[group], 'NX_class', 'HDFGroup'), group)
        if list_datasets:
            out += '\n   '.join(['%s: %s' % (cls, name) for name, cls in zip(names, nxclass)])
        else:
            out += ','.join(names)
        out += '\n'
    return out


def tree_compare(hdf_group1, hdf_group2):
    """
    Compare two hdf groups, display data from both, highlighting differences and missing datasets
    :param hdf_group1: hdf5 File or Group object
    :param hdf_group2: hdf5 File or Group object
    :return: str
    """
    group1_addresses = dataset_addresses(hdf_group1)
    out = ''
    missing_addresses = []
    for address in group1_addresses:
        data_str1 = dataset_string(hdf_group1[address])
        if address in hdf_group2:
            data_str2 = dataset_string(hdf_group2[address])
        else:
            data_str2 = 'Not available'
            missing_addresses += [address]
        diff = '' if data_str1 == data_str2 else '***'
        out += '%60s  %20s  : %20s  %s\n' % (address, data_str1, data_str2, diff)
    out += '\nMissing addresses:\n  '
    out += '\n  '.join(missing_addresses)
    return out


"----------------------ADDRESS DATASET FUNCTIONS------------------------------"


def get_address(hdf_group, name, address_list=None, exact_only=False, return_group=False, find_any=False):
    """
    Return address of dataset that most closely matches str name
     if multiple addresses match, take the longest array
     if name does not match any address, None is returned
    :param hdf_group: hdf5 File or Group object
    :param name: str or list of str of dataset address or name
    :param address_list: list of str of dataset addresses (None to generate from hdf_group)
    :param exact_only: Bool, if True only searches for exact matches to name
    :param return_group: Bool if True returns the group address rather than dataset address
    :param find_any: Bool if True searches for name anywere in address
    :return: str address of best match or list of str address with same length as name list or None if no match
    """
    names = np.asarray(name, dtype=str).reshape(-1)
    if address_list is None:
        address_list = dataset_addresses(hdf_group)
    addresses = []
    for name in names:
        # address
        if is_dataset(hdf_group.get(name)):
            addresses += [name]
            continue
        elif return_group and is_group(hdf_group.get(name)):
            addresses += [name]
            continue

        # search tree
        f_address = find_cascade(name, address_list, exact_only, find_any)
        if not f_address:
            addresses += [None]
            continue

        if return_group:
            f_address = address_group(f_address[0], name)
            addresses += [f_address]
            continue

        # select longest length dataset
        if len(f_address) > 1:
            datasets = [hdf_group.get(ad) for ad in f_address]
            max_len = np.argmax([ds.size for ds in datasets if is_dataset(ds)])
            addresses += [f_address[int(max_len)]]
        else:  # len address == 1
            addresses += f_address

    if len(names) == 1:
        return addresses[0]
    return addresses


def find_nxclass(hdf_group, nxclass='NX_detector'):
    """
    Returns location of hdf group with attribute ['NX_class']== nxclass
    :param hdf_group: hdf5 File or Group object
    :param nxclass: str
    :return: str hdf address
    """
    if 'NX_class' in hdf_group.attrs and hdf_group.attrs['NX_class'] == nxclass.encode():
        return hdf_group.name
    try:
        for branch in hdf_group.keys():
            address = find_nxclass(hdf_group.get(branch), nxclass)
            if address:
                return address
    except AttributeError:
        pass


def find_attr(hdf_group, attr='axes'):
    """
    Returns location of hdf attribute
    Works recursively - starts at the top level and searches all lower hdf groups
    :param hdf_group: hdf5 File or Group object
    :param attr: str : attribute name to search for
    :return: str hdf address
    """
    if attr in hdf_group.attrs:
        attr_names = np.asarray(hdf_group.attrs[attr], dtype=str).reshape(-1)
        address = [hdf_group.get(ax).name for ax in attr_names]
        return address
    try:
        for branch in hdf_group.keys():
            address = find_attr(hdf_group.get(branch), attr)
            if address:
                return address
    except AttributeError:
        pass
    return []


def auto_xyaxis(hdf_group, cmd_string=None, address_list=None):
    """
    Find default axes, signal hdf addresses
    :param hdf_group: hdf5 File or Group object
    :param cmd_string: str of command to take x,y axis from as backup
    :param address_list: list of str of dataset addresses (None to generate from hdf_group)
    :return: xaxis_address, yaxis_address
    """
    try:
        # try fast nexus compliant method
        xaddress, yaddress = nexus_xyaxis(hdf_group)
    except KeyError:
        xaddress = ''
        yaddress = ''
        if cmd_string:
            xname = fn.axes_from_cmd(cmd_string)
            xaddress = get_address(hdf_group, xname, address_list, exact_only=True)
            yname = fn.signal_from_cmd(cmd_string)
            yaddress = get_address(hdf_group, yname, address_list, exact_only=True)

        if not xaddress:
            try:
                xaddress = find_attr(hdf_group, 'axes')[0]
            except IndexError:
                raise KeyError('axes not found in hdf hierachy')
        if not yaddress:
            try:
                yaddress = find_attr(hdf_group, 'signal')[0]
            except IndexError:
                raise KeyError('signal not found in hdf hierachy')
    return xaddress, yaddress


def nexus_xyaxis(hdf_group):
    """
    Nexus compliant method of finding default plotting axes in hdf files
     - find "default" entry in top File group
     - find "default" data in entry
     - find "axes" attr in default data
     - find "signal" attr in default data
     - generate addresses of signal and axes
     if not nexus compliant, raises KeyError
    This method is very fast but only works on nexus compliant files
    :param hdf_group: hdf5 File
    :return axes_address, signal_address: str hdf addresses
    """
    # From: https://manual.nexusformat.org/examples/h5py/index.html
    nx_entry = hdf_group[hdf_group.attrs["default"]]
    nx_data = nx_entry[nx_entry.attrs["default"]]
    axes_list = np.asarray(nx_data.attrs["axes"], dtype=str).reshape(-1)
    signal_list = np.asarray(nx_data.attrs["signal"], dtype=str).reshape(-1)
    axes_address = nx_data[axes_list[0]].name
    signal_address = nx_data[signal_list[0]].name
    return axes_address, signal_address


def badnexus_xyaxis(hdf_group):
    """
    Non-Nexus compliant method of finding default plotting axes in hdf files
     - search hdf hierarchy for attrs "axes" and "signal"
     - generate address of signal and axes
    raises KeyError if axes or signal is not found
    This method can be quite slow but is will work on many old nexus files.
    :param hdf_group: hdf5 File or Group object
    :return axes_address, signal_address: str hdf addresses
    """
    axes_address = find_attr(hdf_group, 'axes')
    signal_address = find_attr(hdf_group, 'signal')
    if len(axes_address) == 0:
        raise KeyError('axes not found in hdf hierachy')
    if len(signal_address) == 0:
        raise KeyError('signal not found in hdf hierachy')
    return axes_address[0], signal_address[0]


def nexus_axes(hdf_group):
    """
    Nexus compliant method of finding default plotting axes in hdf files
     - find "default" entry in top File group
     - find "default" data in entry
     - find "axes" attr in default data
     - generate addresses of axes
     if not nexus compliant, raises KeyError
    This method is very fast but only works on nexus compliant files
    :param hdf_group: hdf5 File
    :return axes_address: str hdf addresses
    """
    # From: https://manual.nexusformat.org/examples/h5py/index.html
    nx_entry = hdf_group[hdf_group.attrs["default"]]
    nx_data = nx_entry[nx_entry.attrs["default"]]
    axes_list = np.asarray(nx_data.attrs["axes"], dtype=str).reshape(-1)
    axes_address = nx_data[axes_list[0]].name
    return axes_address


def nexus_signal(hdf_group):
    """
    Nexus compliant method of finding default plotting axes in hdf files
     - find "default" entry in top File group
     - find "default" data in entry
     - find "signal" attr in default data
     - generate addresses of signal and axes
     if not nexus compliant, raises KeyError
    This method is very fast but only works on nexus compliant files
    :param hdf_group: hdf5 File
    :return signal_address: str hdf addresses
    """
    # From: https://manual.nexusformat.org/examples/h5py/index.html
    nx_entry = hdf_group[hdf_group.attrs["default"]]
    nx_data = nx_entry[nx_entry.attrs["default"]]
    signal_list = np.asarray(nx_data.attrs["signal"], dtype=str).reshape(-1)
    signal_address = nx_data[signal_list[0]].name
    return signal_address


def auto_axes(hdf_group, cmd_string=None, address_list=None, cmd_axes_names=None):
    """
    Find default axes hdf addresses
    :param hdf_group: hdf5 File or Group object
    :param cmd_string: str of command to take x,y axis from as backup
    :param address_list: list of str of dataset addresses (None to generate from hdf_group)
    :param cmd_axes_names: dict of names to pass to axes_from_cmd
    :return: xaxis_address
    """
    try:
        return nexus_axes(hdf_group)
    except KeyError:
        pass

    if cmd_string:
        name = fn.axes_from_cmd(cmd_string, cmd_axes_names)
        address = get_address(hdf_group, name, address_list)
        if address:
            return address

    # cmd failed or not available, look for axes attribute
    address = find_attr(hdf_group, 'axes')
    if address:
        return address[0]

    # axes not in attrs, find first full length 1D array
    if address_list is None:
        address_list = dataset_addresses(hdf_group)

    array_len = np.max([hdf_group.get(adr).size for adr in address_list if hdf_group.get(adr).ndim == 1])
    address = [adr for adr in address_list if hdf_group.get(adr).ndim == 1 and hdf_group.get(adr).size == array_len]
    if address:
        return address[0]
    raise KeyError('axes not found in hdf hierachy')


def auto_signal(hdf_group, cmd_string=None, address_list=None, cmd_signal_names=None):
    """
    Find default signal hdf addresses
    :param hdf_group: hdf5 File or Group object
    :param cmd_string: str of command to take x,y axis from as backup
    :param address_list: list of str of dataset addresses (None to generate from hdf_group)
    :param cmd_signal_names: dict of names to pass to signal_from_cmd
    :return: yaxis_address
    """
    try:
        return nexus_signal(hdf_group)
    except KeyError:
        pass

    if cmd_string:
        name = fn.signal_from_cmd(cmd_string, cmd_signal_names)
        address = get_address(hdf_group, name, address_list)
        if address:
            return address

    # cmd failed or not available, look for signal attribute
    address = find_attr(hdf_group, 'signal')
    if address:
        return address[0]

    # signal not in attrs, find first full length 1D array
    if address_list is None:
        address_list = dataset_addresses(hdf_group)

    array_len = np.max([hdf_group.get(adr).size for adr in address_list if hdf_group.get(adr).ndim == 1])
    address = [adr for adr in address_list if hdf_group.get(adr).ndim == 1 and hdf_group.get(adr).size == array_len]
    if address:
        return address[0]
    raise KeyError('signal not found in hdf hierachy')


"------------------------------------ IMAGE FUNCTIONS  ----------------------------------------------------"


def find_image(hdf_group, address_list=None, multiple=False):
    """
    Return address of image data in hdf file
    Images can be stored as list of file directories when using tif file,
    or as a dynamic hdf link to a hdf file.

    :param hdf_group: hdf5 File or Group object
    :param address_list: list of str: list of str to search in
    :param multiple: if True, return list of all addresses matching criteria
    :return: str or list of str
    """
    if address_list is None:
        address_list = dataset_addresses(hdf_group)
    all_addresses = []
    # First look for 2D image data
    for address in address_list:
        data = hdf_group.get(address)
        if not data or data.size == 1: continue
        if len(data.shape) > 1 and 'signal' in data.attrs:
            if multiple:
                all_addresses += [address]
            else:
                return address
    # Second look for image files
    for address in address_list:
        data = hdf_group.get(address)
        if 'signal' in data.attrs:  # not sure if this generally true, but seems to work for pilatus and bpm images
            if multiple:
                all_addresses += [address]
            else:
                return address
        """
        file = str(data[0])
        file = os.path.join(filepath, file)
        if os.path.isfile(file):
            if multiple:
                all_addresses += [address]
            else:
                return address
        """
    if multiple:
        return all_addresses
    else:
        return None


"----------------------------------------------------------------------------------------------------------------------"
"-------------------------------------------- DatasetWrapper ----------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"


class HdfDataset:
    """
    HDF Dataset reloader
    Self contained holder for a HDF5 dataset, will load the data when called
      dataset = HdfAddress('hdf/file/path.hdf', '/dataset/address')
      data = dataset()
    HdfDataset has attributes:
        dataset.filename
        dataset.address
        dataset.name
        dataset.group
        dataset.size
        dataset.shape
        dataset.ndim
        dataset.len
    If the hdf address doesn't associate with a dataset in the hdf file, a KeyError is raised
    """
    size = 0
    shape = 0
    ndim = 0
    len = 0

    def __init__(self, filename, address):
        self.filename = filename
        self.address = address
        self.name = address_name(address)
        self.group = address_group(address)
        # Check address
        with load(self.filename) as hdf:
            dataset = hdf.get(self.address)
            if dataset is None:
                raise KeyError('"%s" is not availble in %s' % (self.address, self.filename))
            self._update(dataset)

    def __repr__(self):
        return "HdfDataset(\"%s\", \"%s\", shape: %s)" % (self.filename, self.address, self.shape)

    def __len__(self):
        return self.len

    def __call__(self):
        return self._load_data()

    def _update(self, dataset):
        self.size = dataset.size
        self.shape = dataset.shape
        self.ndim = dataset.ndim
        self.len = dataset.len()

    def _load_data(self):
        with load(self.filename) as hdf:
            dataset = hdf.get(self.address)
            self._update(dataset)
            data = dataset_data(dataset)
        return data

    def files(self, filenames, default=None):
        """Generate another address object pointing at a different file"""
        filenames = fn.liststr(filenames)
        if len(filenames) == 1:
            try:
                return HdfDataset(filenames[0], self.address)
            except KeyError:
                return default
        out = []
        for filename in filenames:
            out += [self.files(filename, default)]
        return out

    def dataset(self):
        """Return hdf dataset"""
        hdf = load(self.filename)
        return hdf.get(self.address)

    def data(self):
        """Return data directly from dataset"""
        with load(self.filename) as hdf:
            dataset = hdf.get(self.address)
            self._update(dataset)
            data = dataset_data(dataset)
        return data

    def string(self):
        """Return string from dataset"""
        with load(self.filename) as hdf:
            dataset = hdf.get(self.address)
            self._update(dataset)
            data = dataset_string(dataset)
        return data

    def value(self):
        """Return float value or mean of array"""
        with load(self.filename) as hdf:
            dataset = hdf.get(self.address)
            self._update(dataset)
            data = np.mean(dataset)
        return data

    def array(self, array_len=1):
        """Return array, single values are copied"""
        data = self.data()
        if self.ndim == 1:
            return data
        if self.ndim == 0:
            return np.repeat(data, array_len)
        return np.reshape(data, -1)


"----------------------------------------------------------------------------------------------------------------------"
"---------------------------------------------- HdfWrapper ------------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"


class HdfWrapper(h5py.File):
    """
    Implementation of h5py.File, with additional functions
    nx = Hdf5Nexus('/data/12345.nxs')

    Additional functions:
        nx.nx_dataset_addresses() - list of all hdf addresses for datasets
        nx.nx_tree_str() - string of internal data structure
        nx.nx_find_name('eta') - returns hdf address
        nx.nx_find_addresses( addresses=['/']) - returns list of addresses
        nx.nx_find_attr(attr='signal') - returns address with attribute
        nx.nx_find_image() - returns address of image data
        nx.nx_getdata(address) - returns numpy array of data at address
        nx.nx_array_data(n_points, addresses) - returns dict of n length arrays and dict of addresses
        nx.nx_value_data(addresses) - returns dict of values and dict of addresses
        nx.nx_str_data(addresses, format) - returns dict of string output and dict of addresses
        nx.nx_image_data(index, all) - returns 2/3D array of image data
    """
    def __init__(self, filename, mode='r', *args, **kwargs):
        super(HdfWrapper, self).__init__(filename, mode, *args, **kwargs)

    def nx_reload(self):
        """Closes the hdf file and re-opens"""
        filename = self.filename
        self.close()
        self.__init__(filename)

    def tree(self, address='/', detail=False):
        return tree(self.get(address), detail=detail)

    def dataset_addresses(self, addresses='/', recursion_limit=100, get_size=None, get_ndim=None):
        """
        Return list of addresses of datasets, starting at each address
        :param addresses: list of str or str : time_start in this / these addresses
        :param recursion_limit: Limit on recursivley checking lower groups
        :param get_size: None or int, if int, return only datasets with matching size
        :param get_ndim: None or int, if int, return only datasets with matching ndim
        :return: list of str
        """
        return dataset_addresses(self.get('/'), addresses, recursion_limit, get_size, get_ndim)

    def find(self, name, match_case=True, whole_word=True):
        address_list = self.dataset_addresses()
        return find_name(name, address_list, match_case, whole_word)

    def find_image(self):
        return find_image(self.get('/'), multiple=True)

    def find_attr(self, attr):
        return find_attr(self.get('/'), attr)

    def find_nxclass(self, nx_class):
        return find_nxclass(self.get('/'), nx_class)


"----------------------------------------------------------------------------------------------------------------------"
"---------------------------------------------- HdfScan -------------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"


class HdfScan(Scan):
    """
    Scan for HDF files
    Only reads data when requested, and stores data in the internal namespace
    Data can be requested using the hdf address or the name of the dataset (e.g. /entry1/data/name)
    Usage:
        d = HdfScan('hdf_file.nxs')
        d('entry1/data/sum') >> finds dataset at this location, returns the array
        d('eta') >> finds dataset called 'eta' in hdf file, returns the array
        d.string_format('eta = {eta:5.4f}') >> return formatted string
        d.address('name') >> returns hdf address of 'name'
        d.load_hdf() >> returns h5py like hdf file object with some extra methods
        d.load_all() >> loads all data from file into memory (no lazy loading)
        d.hdf_addresses() >> returns full list of hdf addresses
        d.tree() >> returns str of hdf structure
        d.find_name('name') >> returns list of hdf addresses matching 'name'
        d.find_image() >> returns location of image data, either tiff link or 3d volume
        d.axes() >> automatically finds the default xaxis, returns the array
        d.signal() >> automatically finds the default yaxis, returns the array
        d.image(idx) >> finds the image location if available and returns a detector image
        d.array('name') >> returns array of data item 'name'
        d.value('name') >> returns averaged value of data item 'name'
        d.string('name') >> returns formatted string of data item 'name'
        d.time('name') >> returns datetime object
    """
    def __init__(self, filename, **kwargs):
        self.filename = filename
        self.file = fn.file2name(filename)
        self.scan_number = fn.scanfile2number(filename)
        namespace = {
            'filename': filename,
            'filetitle': self.file,
            'scan_number': self.scan_number
        }
        alt_names = {
            # shortcut: name in namespace
            'scanno': ['scan_number'],
            'cmd': ['scan_command'],
            'energy': ['en'],
        }
        super(HdfScan, self).__init__(namespace, alt_names, **kwargs)

        #self._label_str.extend(['scan_number', 'filetitle'])
        self._hdf_address_list = []
        self._hdf_name2address = {}

    def reset(self):
        """Reset the namespace"""
        self._namespace = {
            'filename': self.filename,
            'filetitle': self.file,
            'scanno': self.scan_number
        }

    def __repr__(self):
        out = 'HdfScan(filename: %s, namespace: %d, associations: %d)'
        return out % (self.filename, len(self._namespace), len(self._alt_names))

    def load_hdf(self):
        """Open and return hdf.File object"""
        return HdfWrapper(self.filename)

    def dataset(self, name):
        """Return dataset object"""
        address = self.address(name)
        return HdfDataset(self.filename, address)

    def tree(self, group_address='/', detail=False, groups=False, recursion_limit=100):
        """
        Return str of the full tree of data in a hdf object
          print(scan.tree())
        ** Note using scan.tree(detail=True) will load all data into memory.
        :param group_address: str address of hdf group to time_start in
        :param detail: False/ True - provide further information about each group and dataset, including attributes
        :param groups: False/ True - only provide group level information
        :param recursion_limit: int max number of levels
        :return: str
        """
        with load(self.filename) as hdf:
            out = tree(hdf[group_address], detail, groups, recursion_limit)
        return out
    info = tree

    def hdf_structure(self, list_datasets=False):
        """Return string displaying the structure of the hdf file"""
        with load(self.filename) as hdf:
            out = nexus_tree(hdf, list_datasets)
        return out

    def hdf_datasets(self):
        """Return string displaying the available datasets in the hdf file"""
        address_list = self._dataset_addresses()
        with load(self.filename) as hdf:
            str_list = [dataset_string(hdf[address]) for address in address_list]
        return '\n'.join(str_list)

    def hdf_compare(self, scan_or_filename):
        """
        Compare hdf tree to another HDF file, display data from both, highlighting differences and missing datasets
        :param scan_or_filename: HdfScan or str filename of hdf file
        :return: str
        """
        if isinstance(scan_or_filename, HdfScan):
            scan_or_filename = scan_or_filename.filename

        out = 'Comparing: %s\n     with: %s\n' % (self.filename, scan_or_filename)
        with load(self.filename) as hdf1, load(scan_or_filename) as hdf2:
            out += tree_compare(hdf1, hdf2)
        return out

    def add2namespace(self, name, data=None, other_names=None, default_value=None, hdf_address=None):
        """
        set data in namespace
        :param name: str name
        :param data: any or None, data to store in namespace (nothing stored if None)
        :param other_names: str, list of str or None - strings to associate with name, giving the same result
        :param default_value: any or None, data to store in default_value namespace (nothing stored if None)
        :param hdf_address: str address in hdf file
        :return: None
        """
        super(HdfScan, self).add2namespace(name, data, other_names, default_value)
        if hdf_address:
            self._hdf_name2address[name] = hdf_address
            self._debug('namespace', 'Add hdf address: %s: %s' % (name, hdf_address))

    def _dataset_addresses(self):
        """
        Return list of hdf addresses in hdf file
        :return: list of str
        """
        if self._hdf_address_list:
            return self._hdf_address_list
        self._debug('hdf', 'Loading address list from %s' % self.file)
        with load(self.filename) as hdf_group:
            out = dataset_addresses(hdf_group)
        self._hdf_address_list = out
        return out

    def _load_data(self, name):
        """
        Load data from hdf file
          Overloads Scan._load_data to read hdf file
          if 'name' not available, raises KeyError
        :param name: str name or address of data
        """
        address_list = self._dataset_addresses()
        # find data address
        with load(self.filename) as hdf:
            address = get_address(hdf, name, address_list)
            self._debug('hdf', 'Search hdf for %s, find: %s' % (name, address))
            if not address and name in self._alt_names:
                for alt_name in self._alt_names[name]:
                    # alt_names must find an exact match
                    address = get_address(hdf, alt_name, address_list, exact_only=True)
                    self._debug('hdf', 'Alt. Search hdf for %s, find: %s' % (alt_name, address))
                    if address is not None: break
            if not address:
                raise KeyError('\'%s\' not available in hdf file' % name)
            dataset = hdf.get(address)
            data = dataset_data(dataset)
        # Store for later use
        self.add2namespace(name, data, address, hdf_address=address)

    def _find_defaults(self):
        """
        Find default axes and signal (x-axis/y-axis), adds to namespace
         Overloads Scan._find_defaults to attempt Nexus compliant defaults
        :return: axes_name, signal_name
        """
        scan_command = self.scan_command()
        address_list = self._dataset_addresses()
        # find data address
        with load(self.filename) as hdf:
            #axes_address, signal_address = auto_xyaxis(hdf, scan_command, address_list)
            axes_address = auto_axes(hdf, scan_command, address_list, self._axes_cmd_names)
            signal_address = auto_signal(hdf, scan_command, address_list, self._axes_cmd_names)
            axes_dataset = hdf.get(axes_address)
            signal_dataset = hdf.get(signal_address)
            # Catch axes being wrong size
            if axes_dataset.size == 1 and signal_dataset.size > 1:
                axes_data = np.arange(len(signal_dataset.size))
            else:
                axes_data = dataset_data(axes_dataset)
            if np.ndim(axes_data) == 0:
                axes_data = np.reshape(axes_data, -1)
            signal_data = dataset_data(signal_dataset)
            if np.ndim(signal_data) == 0:
                signal_data = np.reshape(signal_data, -1)
            axes_name = address_name(axes_address)
            signal_name = address_name(signal_address)
        self.add2namespace(axes_name, axes_data, self._axes_str, hdf_address=axes_address)
        self.add2namespace(signal_name, signal_data, self._signal_str, hdf_address=signal_address)
        return axes_name, signal_name

    def load_all(self):
        """
        Loads all hdf data into memory
        :return: None
        """
        address_list = self._dataset_addresses()
        # find data address
        with load(self.filename) as hdf:
            for address in address_list:
                name = address_name(address)
                dataset = hdf.get(address)
                data = dataset_data(dataset)
                self.add2namespace(name, data, address, hdf_address=address)

    def hdf_addresses(self):
        """
        Return list of all hdf addresses
        :return: list(str)
        """
        return self._dataset_addresses()

    def address(self, name):
        """
        Return hdf address of namespace name
        :param name: str name in namespace
        :return: str hdf address
        """
        if name in self._hdf_name2address:
            return self._hdf_name2address[name]
        if name in self._alt_names:
            for alt_name in self._alt_names[name]:
                if alt_name in self._hdf_name2address:
                    return self._hdf_name2address[alt_name]
        self._load_data(name)
        return self._hdf_name2address[name]

    def group_addresses(self, group_name):
        """
        Return list of hdf addresses
        :param group_name: str name of hdf grop
        :return: list of str
        """
        addresses = self._dataset_addresses()
        group_name = '/%s/' % group_name
        return [adr for adr in addresses if group_name in adr]

    def find_address(self, name, match_case=False, whole_word=False):
        """
        Find datasets using field name
        :param name: str : name to match in dataset field name
        :param match_case: if True, match case of name
        :param whole_word: if True, only return whole word matches
        :return: list of str matching dataset addresses
        """
        address_list = self._dataset_addresses()
        return find_name(name, address_list, match_case, whole_word)

    def find_image(self, multiple=False):
        """
        Return address of image data in hdf file
        Images can be stored as list of file directories when using tif file,
        or as a dynamic hdf link to a hdf file.
        :param multiple: if True, return list of all addresses matching criteria
        :return: str or list of str of hdf addreses
        """
        address_list = self._dataset_addresses()
        with load(self.filename) as hdf:
            out = find_image(hdf, address_list, multiple)
        return out

    def _set_volume(self, array=None, image_file_list=None, image_address=None, hdf_file=None):
        """
        Set the scan file volume
        :param array: None or [scan_len, i, j] size array
        :param image_file_list: list of str path locations for [scan_len] image files
        :param image_address: str hdf address of image location
        :param hdf_file: str path location of a hdf file (None to use current file)
        :return: None, sets self._volume
        """
        if hdf_file is None:
            hdf_file = self.filename
        
        if image_file_list is None and image_address is not None:
            hdf = load(hdf_file)
            image_address = get_address(hdf, image_address)
            dataset = hdf.get(image_address)

            # if array - return array
            if dataset.ndim == 3:
                # array data in hdf dataset
                self._volume = DatasetVolume(dataset)
                return
            elif dataset.ndim >= 3:
                # multi-dimensional scan, reshape to single scan dimension
                array = np.reshape(dataset, (-1, dataset.shape[-2], dataset.shape[-1]))
            else:
                # image_address points to a list of filenames
                # image_file_list = [fn.bytestr2str(file) for file in dataset]
                image_file_list = np.reshape(dataset, -1).astype(str)  # handle multi-dimensional arrays
            hdf.close()
        
        if image_file_list is not None:
            # e.g. list of tiff files
            image_file_list = fn.liststr(image_file_list)
            # Check filenames
            if not os.path.isfile(image_file_list[0]):
                # filename maybe absolute, just take the final folder
                abs_filepath = os.path.dirname(self.filename)
                f = ['/'.join(os.path.abspath(filename).replace('\\', '/').split('/')[-2:]) for filename in
                     image_file_list]
                image_file_list = [os.path.join(abs_filepath, file) for file in f]

        super(HdfScan, self)._set_volume(array, image_file_list)

    def volume(self, image_address=None, hdf_file=None, image_file_list=None, array=None):
        """
        Load image from hdf file, works with either image addresses or stored arrays
        :param image_address: str hdf address of image location
        :param hdf_file: str path location of a hdf file (None to use current file)
        :param image_file_list: list of str path locations for [scan_len] image files
        :param array: None or [scan_len, i, j] size array
        :return: ImageVolume or HdfVolume
        """
        if self._volume and image_address is None and image_file_list is None and array is None:
            return self._volume
        if hdf_file:
            self._image_name = image_address
        elif image_address:
            image_address = self.address(image_address)
            self._image_name = image_address
        elif self._image_name:
            image_address = self._image_name
        else:
            image_address = self.find_image()
            if not image_address:
                raise KeyError('image path template not found in %r' % self)
            self._image_name = image_address

        self._set_volume(array, image_file_list, image_address, hdf_file)
        return self._volume

    def _prep_operation(self, operation):
        """
        prepare operation string, replace names with names in namespace
          Overloaded from Scan method, runs beforehand and replaces hdf addresses
        :param operation: str
        :return operation: str, names replaced to match namespace
        """

        old_op = operation
        # First look for addresses in operation to seperate addresses from divide operations
        addresses = fn.re_address.findall(operation)
        ds_addresses = self._dataset_addresses()
        for address in addresses:
            if address in ds_addresses:
                operation = operation.replace(address, address_name(address))
        self._debug('eval', 'Prepare eval operation for HDF\n  initial: %s\n  final: %s' % (old_op, operation))
        return super(HdfScan, self)._prep_operation(operation)
