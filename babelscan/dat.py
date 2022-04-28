"""
Functions for reading .dat files
"""

import os
import numpy as np
from collections import OrderedDict

from . import functions as fn
from .babelscan import Scan
from .volume import ImageVolume


"----------------------------LOAD FUNCTIONS---------------------------------"


class Dict2Obj(OrderedDict):
    """
    Convert dictionary object to class instance
    """

    def __init__(self, dictvals, order=None):
        super(Dict2Obj, self).__init__()

        if order is None:
            order = dictvals.keys()

        for name in order:
            setattr(self, name, dictvals[name])
            self.update({name: dictvals[name]})


def read_dat_file(filename):
    """
    Reads #####.dat files from instrument, returns class instance containing all data
    Input:
      filename = string filename of data file
    Output:
      d = class instance with parameters associated to scanned values in the data file, plus:
         d.metadata - class containing all metadata from datafile
         d.keys() - returns all parameter names
         d.values() - returns all parameter values
         d.items() - returns parameter (name,value) tuples
    """
    f = open(filename, 'r')
    lines = f.readlines()
    f.close()

    # Read metadata
    meta = OrderedDict()
    lineno = 0
    for ln in lines:
        lineno += 1
        if '&END' in ln: break
        ln = ln.strip(' ,\n')
        neq = ln.count('=')
        if neq == 1:
            'e.g. cmd = "scan x 1 10 1"'
            inlines = [ln]
        elif neq > 1:
            'e.g. SRSRUN=571664,SRSDAT=201624,SRSTIM=183757'
            inlines = ln.split(',')
        else:
            'e.g. <MetaDataAtStart>'
            continue

        for inln in inlines:
            vals = inln.split('=')
            if len(vals) != 2: continue
            try:
                meta[vals[0]] = eval(vals[1])
            except:
                meta[vals[0]] = vals[1]

    # Read Main data
    # previous loop ended at &END, now starting on list of names
    names = lines[lineno].split()
    # Load 2D arrays of scanned values
    vals = np.loadtxt(lines[lineno + 1:], ndmin=2)
    # Assign arrays to a dictionary
    main = OrderedDict()
    for name, value in zip(names, vals.T):
        main[name] = value

    # Convert to class instance
    d = Dict2Obj(main, order=names)
    d.metadata = Dict2Obj(meta)
    return d


"----------------------------------------------------------------------------------------------------------------------"
"---------------------------------------------- DatScan -------------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"


class DatScan(Scan):
    """
    Scan for .dat files
    Reads data into babelscan class, storing data in the internal namespace
    Scan data and metadata can be requested using the the name of the dataset (e.g. 'eta')
    Usage:
        d = DatScan('file.dat')
        d('eta') >> finds data column or metadata called 'eta', returns the array
        d.axes() >> automatically finds the default xaxis, returns the array
        d.signal() >> automatically finds the default yaxis, returns the array
        d.image(idx) >> finds the image location if available and returns a detector image
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
            # shortcut: name in file
            'scanno': ['scan_number'],
            'scan_command': ['cmd'],
            'energy': ['en'],
        }
        super(DatScan, self).__init__(namespace, alt_names, **kwargs)
        #self._label_str.extend(['scanno', 'filetitle'])

    def reset(self):
        """Reset the namespace"""
        self._namespace = {
            'filename': self.filename,
            'scanno': self.scan_number
        }

    def __repr__(self):
        out = 'DatScan(filename: %s, namespace: %d, associations: %d)'
        return out % (self.filename, len(self._namespace), len(self._alt_names))

    def _load_data(self, name):
        """
        Load data from hdf file
          Overloads Scan._load_data to read hdf file
          if 'name' not available, raises KeyError
        :param name: str name or address of data
        """
        dataobj = read_dat_file(self.filename)
        self._namespace.update(dataobj.metadata)
        self._namespace.update(dataobj)
        self.add2namespace('_scan_data', dataobj)
        if name in self._namespace:
            return
        if name in self._alt_names:
            for alt_name in self._alt_names[name]:
                if alt_name in self._namespace:
                    return
        super(DatScan, self)._load_data(name)

    def _find_defaults(self):
        """
        Find default axes and signal (x-axis/y-axis), adds to namespace
         Overloads Scan._find_defaults to fall back on first/ last item in list
        :return: axes_name, signal_name
        """
        scan_command = self.scan_command()
        # axes / x-axis
        axes_name = fn.axes_from_cmd(scan_command, self._axes_cmd_names)
        try:
            axes_data = self._get_data(axes_name)
        except KeyError:
            scan_data = self._get_data('_scan_data')
            axes_name = list(scan_data.keys())[0]  # make use of ordered dict
            axes_data = self._get_data(axes_name)
        self.add2namespace(axes_name, axes_data, self._axes_str)
        # signal / y-axis
        signal_name = fn.signal_from_cmd(scan_command, self._signal_cmd_names)
        try:
            signal_name, signal_data = self._get_name_data(signal_name)
        except KeyError:
            scan_data = self._get_data('_scan_data')
            signal_name = list(scan_data.keys())[-1]  # make use of ordered dict
            signal_data = self._get_data(signal_name)
        self.add2namespace(signal_name, signal_data, self._signal_str)
        return axes_name, signal_name

    def tree(self):
        """Return str of data in dat file"""
        s = "%r\nMetadata:\n" % self
        scan_data = self._get_data('_scan_data')
        s += '\n'.join(self.string(list(scan_data.metadata.keys())))
        s += "\nScan data:\n"
        s += '\n'.join(self.string(list(scan_data.keys())))
        return s
    info = tree

    def _set_volume(self, array=None, image_file_list=None):
        """
        Set the scan file volume
        :param array: None or [scan_len, i, j] size array
        :param image_file_list: list of str path locations for [scan_len] image files
        :return: None, sets self._volume
        """

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

        super(DatScan, self)._set_volume(array, image_file_list)

    def volume(self, path_template=None, image_file_list=None, array=None):
        """
        Load image as ImageVolume
        :param path_template: str template of detector images e.g. 'folder/%d.tiff'
        :param image_file_list: list of str path locations for [scan_len] image files
        :param array: None or [scan_len, i, j] size array
        :return: ImageVolume
        """
        if self._volume and image_file_list is None and array is None:
            return self._volume
        if path_template is None:
            path_template = self._get_data("_path_template")  # image folder path tempalte e.g. folder/%d.tiff
        pointers = self._get_data('path')  # list of image numbers in dat files

        filenames = [path_template % pointer for pointer in pointers]
        self._set_volume(image_file_list=filenames)
        return self._volume
