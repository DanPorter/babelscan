"""
Lazy Volume class
"""

import re
import operator
import numpy as np
import h5py
from imageio import imread

from .settings import init_vol_plot_manager

"----------------------------------------------------------------------------------------------------------------------"
"------------------------------------------- Volume Functions ---------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"

re_findint = re.compile(r'\d+')


def pixel_peak_search(data, peak_percentile=99):
    """
    find average position of bright points in image
    :param data: numpy array with ndims 1,2,3
    :param peak_percentile: float from 0-100, percentile of image to use as peak area
    :return: i, j, k index of image[i,j,k]
    """
    # bright = data > (peak_percentile/100.) * np.max(image)
    bright = data > np.percentile(data, peak_percentile)
    weights = data[bright]

    if np.ndim(data) == 3:
        shi, shj, shk = data.shape
        j, i, k = np.meshgrid(range(shj), range(shi), range(shk))
        avi = np.average(i[bright], weights=weights)
        avj = np.average(j[bright], weights=weights)
        avk = np.average(k[bright], weights=weights)
        return int(avi), int(avj), int(avk)
    elif np.ndim(data) == 2:
        shi, shj = data.shape
        j, i = np.meshgrid(range(shj), range(shi))
        avi = np.average(i[bright], weights=weights)
        avj = np.average(j[bright], weights=weights)
        return int(avi), int(avj)
    elif np.ndim(data) == 1:
        i = np.arange(len(data))
        avi = np.average(i[bright], weights=weights)
        return int(avi)
    else:
        raise TypeError('wrong data type')


def roi(volume, cen_h=None, cen_v=None, wid_h=31, wid_v=31):
    """
    Create new region of interest from detector images
    :param volume: Volume object or numpy.array with ndim==3
    :param cen_h: int or None
    :param cen_v: int or None
    :param wid_h:  int or None
    :param wid_v:  int or None
    :return: l*wid_v*wid_h array
    """
    shape = np.shape(volume)
    if cen_h is None:
        cen_h = shape[2] // 2
    if cen_v is None:
        cen_v = shape[1] // 2
    return volume[:, cen_v - wid_v // 2:cen_v + wid_v // 2, cen_h - wid_h // 2:cen_h + wid_h // 2]


def roi_sum(volume, cen_h=None, cen_v=None, wid_h=31, wid_v=31):
    """
    Create new region of interest from detector images, return sum and max of each image
    :param volume: Volume object or numpy.array with ndim==3
    :param cen_h: int or None
    :param cen_v: int or None
    :param wid_h:  int or None
    :param wid_v:  int or None
    :return: roi_sum, roi_max
    """
    shape = np.shape(volume)
    if cen_h is None:
        cen_h = shape[2] // 2
    if cen_v is None:
        cen_v = shape[1] // 2
    sum_vals = np.array(
        [np.sum(im[cen_v - wid_v // 2:cen_v + wid_v // 2, cen_h - wid_h // 2:cen_h + wid_h // 2]) for im in volume]
    )
    max_vals = np.array(
        [np.max(im[cen_v - wid_v // 2:cen_v + wid_v // 2, cen_h - wid_h // 2:cen_h + wid_h // 2]) for im in volume]
    )
    return sum_vals, max_vals


def check_roi_op(volume, operation):
    """
    Create new region of interest (roi) values from operation string
    The roi centre and size is defined by an operation:
      operation = 'nroi[210, 97, 75, 61]'
      'nroi'      -   creates a region of interest in the detector centre with size 31x31
      'nroi[h,v]' -   creates a roi in the detector centre with size hxv, where h is horizontal, v is vertical
      'nroi[m,n,h,v] - create a roi with cen_h, cen_v, wid_h, wid_v = n, m, h, v
    :param volume: Volume object or numpy.array with ndim==3
    :param operation: str : operation string
    :return: cen_h, cen_v, wid_h, wid_v, operation
    """
    vals = [int(val) for val in re_findint.findall(operation)]
    nvals = len(vals)
    shape = np.shape(volume)
    if 'peak' in operation:
        i, j, k = pixel_peak_search(volume)
        cen_h, cen_v = k, j
    else:
        cen_h, cen_v = shape[2] // 2, shape[1] // 2
    if nvals == 4:
        cen_h, cen_v, wid_h, wid_v = vals
    elif nvals == 2:
        wid_h, wid_v = vals
    else:
        wid_h, wid_v = 31, 31
    operation = 'nroi[%d,%d,%d,%d]' % (cen_h, cen_v, wid_h, wid_v)
    return cen_h, cen_v, wid_h, wid_v, operation


def roi_op(volume, operation):
    """
    Create new region of interest (roi) values from operation string
    The roi centre and size is defined by an operation:
      operation = 'nroi[210, 97, 75, 61]'
      'nroi'      -   creates a region of interest in the detector centre with size 31x31
      'nroi[h,v]' -   creates a roi in the detector centre with size hxv, where h is horizontal, v is vertical
      'nroi[m,n,h,v] - create a roi with cen_h, cen_v, wid_h, wid_v = n, m, h, v
    :param volume: Volume object or numpy.array with ndim==3
    :param operation:  str : operation string
    :return: l*wid_v*wid_h array
    """
    cen_h, cen_v, wid_h, wid_v, operation = check_roi_op(volume, operation)
    return roi(volume, cen_h, cen_v, wid_h, wid_v)


def roi_op_sum(volume, operation):
    """
    Create new region of interest (roi) values from operation string
    The roi centre and size is defined by an operation:
      operation = 'nroi[210, 97, 75, 61]'
      'nroi'      -   creates a region of interest in the detector centre with size 31x31
      'nroi[h,v]' -   creates a roi in the detector centre with size hxv, where h is horizontal, v is vertical
      'nroi[m,n,h,v] - create a roi with cen_h, cen_v, wid_h, wid_v = n, m, h, v
    :param volume: Volume object or numpy.array with ndim==3
    :param operation:  str : operation string
    :return: roi_sum, roi_max
    """
    cen_h, cen_v, wid_h, wid_v, operation = check_roi_op(volume, operation)
    return roi_sum(volume, cen_h, cen_v, wid_h, wid_v)


def rebin(volume, step=(1, 1, 1), average_points=True):
    """
    Rebins the volume along 3 dimensions, taking the average of elements within each step
    :param volume: Volume object or numpy.array with ndim==3
    :param step: [i', j', k'] step size along each dimension
    :param average_points: Bool, if True, averages the points along each step, otherwise just sum
    :return: volume[i//i',j//j',k//k']
    """

    i, j, k = volume.shape

    # remove trailing elements
    vol2 = volume[:i - i % step[0], :j - j % step[1], :k - k % step[2]]

    # sum pixels
    count = 0.0
    volsum = np.zeros(np.floor_divide(vol2.shape, step))
    for n in range(step[0]):
        for m in range(step[1]):
            for o in range(step[2]):
                count += 1.0
                volsum += vol2[n::step[0],m::step[1],o::step[2]]
    if average_points:
        return volsum / count  # Take the average
    return volsum


"----------------------------------------------------------------------------------------------------------------------"
"-------------------------------------------- Volume Classes ----------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"


class Volume:
    """
    Volume functions
      Contains various functions to operate on 3D volumes
    """

    shape = (0, 0)
    size = 0
    ndim = 3

    def _lazy_op(self, operation, other):
        """Lazy operation"""
        if issubclass(type(other), float) or issubclass(type(other), int):
            # compare with value
            return np.array([operation(self.__getitem__(idx), other) for idx in range(self.shape[0])])
        else:
            raise TypeError('ArrayVolume %s %s not implemented yet' % (operation.__name__, type(other)))

    def __getitem__(self, item):
        """Overloaded method"""
        pass

    def __lt__(self, other):
        """self < other"""
        return self._lazy_op(operator.lt, other)

    def __le__(self, other):
        """self <= other"""
        return self._lazy_op(operator.le, other)

    def __gt__(self, other):
        """self > other"""
        return self._lazy_op(operator.gt, other)

    def __ge__(self, other):
        """self > other"""
        return self._lazy_op(operator.ge, other)

    def __eq__(self, other):
        """self == other"""
        return self._lazy_op(operator.eq, other)

    def __ne__(self, other):
        """self != other"""
        return self._lazy_op(operator.ne, other)

    def cut(self, index1=(0, 0, 0), index2=(-1, -1, -1), *args, **kwargs):
        """
        generate arbitary cut between two points in the volume
        :param index1: (i,j,k) start point in the volume
        :param index2: (i,j,k) end point in the volume
        :return: array, length=max difference between index1 and index2
        """

        index1 = np.asarray(index1, dtype=int)
        index2 = np.asarray(index2, dtype=int)
        # assign end values (-1)
        sh = np.array(self.shape)
        index1[index1 < 0] = sh[index1 < 0] + index1[index1 < 0]
        index2[index2 < 0] = sh[index2 < 0] + index2[index2 < 0]
        diff = index2 - index1
        mx = np.max(diff)
        indexes = [
            [index1[i] + n * diff[i] // mx for i in range(self.ndim)]
            for n in range(mx)
        ]
        try:
            return self[indexes]  # uses numpy advanced indexing
        except IndexError:
            return [self[idx[0], idx[1], idx[2]] for idx in indexes]

    def rebin(self, step=(1, 1, 1), average_points=True):
        """
        Rebins the volume along 3 dimensions, taking the average of elements within each step
        :param step: [i', j', k'] step size along each dimension
        :param average_points: Bool, if True, averages the points along each step, otherwise just sum
        :return: volume[i//i',j//j',k//k']
        """
        return rebin(self, step, average_points)

    def sum_flat(self):
        """Return sum of images: np.sum(volume, axis=0)"""
        return np.sum(self, axis=0)

    def array_sum(self):
        """Returns [sum(image) for image in volume]"""
        return np.array([np.sum(im) for im in self])

    def array_max(self):
        """Returns [max(image) for image in volume]"""
        return np.array([np.max(im) for im in self])

    def array_min(self):
        """Returns [min(image) for image in volume]"""
        return np.array([np.min(im) for im in self])

    def array_mean(self):
        """Returns [mean(image) for image in volume]"""
        return np.array([np.mean(im) for im in self])

    def array_argmax(self):
        """Returns [i,j = argmax(im) for im in self]"""
        return np.array([np.unravel_index(np.argmax(im), im.shape) for im in self])

    def array_peak(self):
        """Returns [i,j = peak_search(im) for im in self]"""
        return np.array([pixel_peak_search(im) for im in self])

    def argmax(self):
        """Numpy argmax, return i,j,k"""
        idx = np.argmax(self)
        return np.unravel_index(idx, self.shape)

    def peak_search(self, peak_percentile=99):
        """
        find average position of bright points in image
        :param peak_percentile: float from 0-100, percentile of image to use as peak area
        :return: i, j, k index of image[i,j,k]
        """
        shi, shj, shk = self.shape
        j, i, k = np.meshgrid(range(shj), range(shi), range(shk))
        # bright = image > (peak_percentile/100.) * np.max(image)
        bright = self > np.percentile(self, peak_percentile)
        weights = self[bright]
        avi = np.average(i[bright], weights=weights)
        avj = np.average(j[bright], weights=weights)
        avk = np.average(k[bright], weights=weights)
        return int(avi), int(avj), int(avk)

    def background(self, background_percentile=50):
        """
        Average bottom percentile of elements to determine background value
        :param background_percentile: int from 0-100, percentile of volume to use as background
        :return: float
        """
        bkg_index = self <= np.percentile(self, background_percentile)
        return np.mean(self[bkg_index])

    def peak_amplitude(self, background_percentile=50):
        """
        Sum the volume with background removed
        :param background_percentile: int from 0-100, percentile of volume to use as background
        :return: float
        """
        bkg = self.background(background_percentile)
        idx = self > bkg
        return np.sum(self[idx])

    def roi(self, cen_h=None, cen_v=None, wid_h=31, wid_v=31):
        """
        Create new region of interest from detector images
        :param cen_h: int or None
        :param cen_v: int or None
        :param wid_h:  int
        :param wid_v:  int
        :return: l*wid_v*wid_h array
        """
        if cen_h is None:
            cen_h = self.shape[2] // 2
        if cen_v is None:
            cen_v = self.shape[1] // 2
        return ArrayVolume(self[:, cen_v - wid_v // 2:cen_v + wid_v // 2, cen_h - wid_h // 2:cen_h + wid_h // 2])

    def roi_sum(self, cen_h=None, cen_v=None, wid_h=31, wid_v=31):
        """
        Create new region of interest from detector images, return sum and max of each image
        :param cen_h: int or None
        :param cen_v: int or None
        :param wid_h:  int
        :param wid_v:  int
        :return: roi_sum, roi_max
        """
        if cen_h is None:
            cen_h = self.shape[2] // 2
        if cen_v is None:
            cen_v = self.shape[1] // 2
        sum_vals = np.array(
            [np.sum(im[cen_v - wid_v // 2:cen_v + wid_v // 2, cen_h - wid_h // 2:cen_h + wid_h // 2]) for im in self]
        )
        max_vals = np.array(
            [np.max(im[cen_v - wid_v // 2:cen_v + wid_v // 2, cen_h - wid_h // 2:cen_h + wid_h // 2]) for im in self]
        )
        return sum_vals, max_vals

    def check_roi_op(self, operation):
        """
        Create new region of interest (roi) values from operation string
        The roi centre and size is defined by an operation:
          operation = 'nroi[210, 97, 75, 61]'
          'nroi'      -   creates a region of interest in the detector centre with size 31x31
          'nroi[h,v]' -   creates a roi in the detector centre with size hxv, where h is horizontal, v is vertical
          'nroi[m,n,h,v] - create a roi with cen_h, cen_v, wid_h, wid_v = n, m, h, v
        :param operation: str : operation string
        :return: cen_h, cen_v, wid_h, wid_v, operation
        """
        vals = [int(val) for val in re_findint.findall(operation)]
        nvals = len(vals)
        shape = self.shape
        if 'peak' in operation:
            i, j, k = self.peak_search()
            cen_h, cen_v = k, j
        else:
            cen_h, cen_v = shape[2] // 2, shape[1] // 2
        if nvals == 4:
            cen_h, cen_v, wid_h, wid_v = vals
        elif nvals == 2:
            wid_h, wid_v = vals
        else:
            wid_h, wid_v = 31, 31
        operation = 'nroi[%d,%d,%d,%d]' % (cen_h, cen_v, wid_h, wid_v)
        return cen_h, cen_v, wid_h, wid_v, operation

    def roi_op(self, operation):
        """
        Create new region of interest (roi) from image data and return sum and maxval
        The roi centre and size is defined by an operation:
          operation = 'nroi[210, 97, 75, 61]'
          'nroi'      -   creates a region of interest in the detector centre with size 31x31
          'nroi[h,v]' -   creates a roi in the detector centre with size hxv, where h is horizontal, v is vertical
          'nroi[m,n,h,v] - create a roi with cen_h, cen_v, wid_h, wid_v = n, m, h, v
        :param operation: str : operation string
        :return: l*wid_v*wid_h array
        """
        cen_h, cen_v, wid_h, wid_v, operation = self.check_roi_op(operation)
        return self.roi(cen_h, cen_v, wid_h, wid_v)

    def roi_op_sum(self, operation):
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
        cen_h, cen_v, wid_h, wid_v, operation = self.check_roi_op(operation)
        return self.roi_sum(cen_h, cen_v, wid_h, wid_v)


class ArrayVolume(Volume):
    """
    ArrayVolume for 3D Numpy arrays
      Contains additional functions for 3D arrays
    Usage:
      array = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
      lzvol = ArrayVolume(array)
      image = lzvol[0]
    Supported indexing:
     single dimension indexing, as numpy array:
      vol = lzvol[:3]
      vol = lzvol[slice(1,-1,2)]
     multi-dimension indexing, as numpy array:
      vol = lzvol[3, 100:200]
      vol = lzvol[1:-1, 100:200, 100:200]
     array operations
      len(lzvol), np.shape(lzvol), np.size(lzvol), np.ndim(lzvol)
      np.sum(lzvol), np.mean(lzvol), np.percentile(lzvol)
     boolean array operations
      lzvol > 1 (<, <=, >, >=, ==, !=)
    """

    def __init__(self, array):
        if array.ndim != 3:
            raise TypeError('%r is not a volume' % array)
        self.dataset = array
        self.shape = array.shape
        self.size = array.size
        self.ndim = array.ndim
        self.plot = init_vol_plot_manager(self)

    def __repr__(self):
        return 'ArrayVolume(%r, shape=%s)' % (type(self.dataset), self.shape)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset.__getitem__(item)


class ImageVolume(Volume):
    """
    ImageVolume for images
      Only loads images when called, reducing memory requirements
    Usage:
      lzvol = ImageVolume([file1.tiff, file2.tiff, file3.tiff,...])
      image = lzvol[0]
    Supported indexing:
     single dimension indexing, as numpy array:
      vol = lzvol[:3]
      vol = lzvol[slice(1,-1,2)]
     multi-dimension indexing, as numpy array:
      vol = lzvol[3, 100:200]
      vol = lzvol[1:-1, 100:200, 100:200]
     array operations
      len(lzvol), np.shape(lzvol), np.size(lzvol), np.ndim(lzvol)
      np.sum(lzvol), np.mean(lzvol), np.percentile(lzvol)
     boolean array operations
      lzvol > 1 (<, <=, >, >=, ==, !=)
    """
    def __init__(self, list_of_files):
        self.files = np.asarray(list_of_files, dtype=str).reshape(-1)
        image = self._read_image(0)
        im_shape = np.shape(image)
        self.shape = (len(self.files), im_shape[0], im_shape[1])
        self.size = len(self.files) * np.size(image)
        self.ndim = 3
        self.plot = init_vol_plot_manager(self)

    def __repr__(self):
        return 'ImageVolume([%s,...], shape=%s)' % (self.files[0], self.shape)

    def __len__(self):
        return len(self.files)

    def _read_image(self, idx, slice_i=slice(None), slice_j=slice(None)):
        files = self.files[idx]
        if np.ndim(slice_i) == 2:
            im_op = slice_i
        else:
            im_op = slice_i, slice_j
        if issubclass(type(files), str):
            return imread(files)[im_op]
        return np.array([imread(file)[im_op] for file in files])

    def __getitem__(self, item=None):
        if item is None:
            idx = len(self.files) // 2
            return self._read_image(idx)
        elif type(item) is tuple:
            # multidimensional. items seperated by ,
            if len(item) == 0:
                return self.__getitem__(None)
            elif len(item) == 1:
                return self._read_image(item[0])
            else:
                return self._read_image(item[0], *item[1:])
        elif np.ndim(item) == 3:
            # Boolean mask array, e.g. vol[vol > 1]
            return np.concatenate([self._read_image(i, im) for i, im in enumerate(item)])
        else:
            return self._read_image(item)


class DatasetVolume(h5py.Dataset, Volume):
    """
    DatasetVolume for 3D HDF datasets
      Only loads images when called, reducing memory requirements
    Usage:
      dataset = hdf['/entry/group/name']
      lzvol = DatasetVolume(dataset)
      image = lzvol[0]
    Supported indexing:
     single dimension indexing, as numpy array:
      vol = lzvol[:3]
      vol = lzvol[slice(1,-1,2)]
     multi-dimension indexing, as numpy array:
      vol = lzvol[3, 100:200]
      vol = lzvol[1:-1, 100:200, 100:200]
     array operations
      len(lzvol), np.shape(lzvol), np.size(lzvol), np.ndim(lzvol)
      np.sum(lzvol), np.mean(lzvol), np.percentile(lzvol)
     boolean array operations
      lzvol > 1 (<, <=, >, >=, ==, !=)
    """
    def __init__(self, dataset):
        if dataset.ndim != 3:
            raise TypeError('%r is not a volume as ndim=%d' % (dataset, dataset.ndim))
        super(DatasetVolume, self).__init__(dataset.id)
        self.plot = init_vol_plot_manager(self)

    def __repr__(self):
        return 'DatasetVolume(%r)' % (super(DatasetVolume, self).__repr__())
