"""
Module of General functions
"""

import os
import re
import json
import datetime
from dateutil.parser import parse
import numpy as np


BYTES_DECODER = 'utf-8'
VALUE_FUNCTION = np.mean  # lambda a: np.asarray(a).reshape(-1)[0]
MAX_STRING_LENGTH = 100
VALUE_FORMAT = '%.5g'
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S.%f%z"
OUTPUT_FORMAT = '%20s = %s'
CONFIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'config_files')

# Compile useful pattern strings
re_integer = re.compile(r'\d+')
re_address = re.compile(r'\w*?/[\w/]*')
re_nroi = re.compile(r'nroi.*\[[\d\s,]+\]')  # catch nroi[31, 31], nroi_bkg[243,97,31,41]
re_varname = re.compile(r'[a-zA-Z]\w*\[.+\]|[a-zA-Z]\w*')  # catch ab5, ab[31, 31]
re_strop = re.compile(r'\{(.+?)\}')


def file2name(filename):
    """
    Extracts filename from file
    :param filename: str
    :return: str
    """
    return os.path.basename(filename)


def file2foldername(filename):
    """
    Extracts folder name from file
    :param filename: str
    :return: str
    """
    return os.path.basename(os.path.dirname(os.path.abspath(filename)))


def scanfile2number(filename):
    """
    Extracts the scan number from a .nxs filename
    :param filename: str : filename of .nxs file
    :return: int : scan number
    """
    nameext = os.path.split(filename)[-1]
    name = os.path.splitext(nameext)[0]
    numbers = re_integer.findall(name)
    if len(numbers) > 0:
        return int(numbers[-1])
    return 0


def scannumber2file(number, name_format="%d.nxs"):
    """Convert number to scan file using format"""
    return name_format % number


def shortstr(string):
    """
    Shorten string by removing long floats
    :param string: string, e.g. '#810002 scan eta 74.89533603616637 76.49533603616636 0.02 pil3_100k 1 roi2'
    :return: shorter string, e.g. '#810002 scan eta 74.895 76.495 0.02 pil3_100k 1 roi2'
    """
    #return re.sub(r'(\d\d\d)\d{4,}', r'\1', string)
    def subfun(m):
        return str(round(float(m.group()), 3))
    return re.sub(r'\d+\.\d{5,}', subfun, string)


def bytestr2str(string):
    """
    Convert bytestr or str to str
    Note - providing an array returns a string array
    :param string: Bytes or str
    :return: str
    """
    return np.asarray(string, dtype=str)[()]


def liststr(string):
    """
    Convert str or list of str to list of str
    :param string: str, byteString, list, array
    :return: list of str
    """
    return list(np.asarray(string, dtype=str).reshape(-1))


def data_string(data):
    """
    Return string depiction of data
    :param data: any
    :return: str
    """
    size = np.size(data)
    shape = np.shape(data)
    if size == 1:
        out_str = shortstr(bytestr2str(np.reshape(data, -1)[0]))
        if len(out_str) > MAX_STRING_LENGTH:
            out_str = "%s ..." % out_str[:MAX_STRING_LENGTH]
        return out_str
    try:
        amax = np.max(data)
        amin = np.min(data)
        amean = np.mean(data)
        out_str = "%s max: %4.5g, min: %4.5g, mean: %4.5g"
        return out_str % (shape, amax, amin, amean)
    except TypeError:
        # list of str
        array = np.asarray(data).reshape(-1)
        array_start = array[0]
        array_end = array[-1]
        out_str = "%s [%s, ..., %s]"
        return out_str % (shape, array_start, array_end)


def findranges(scannos, sep=':'):
    """
    Convert a list of numbers to a simple string
    E.G.
    findranges([1,2,3,4,5]) = '1:5'
    findranges([1,2,3,4,5,10,12,14,16]) = '1:5,10:2:16'
    """

    scannos = np.sort(scannos).astype(int)

    dif = np.diff(scannos)

    stt, stp, rng = [scannos[0]], [dif[0]], [1]
    for n in range(1, len(dif)):
        if scannos[n + 1] != scannos[n] + dif[n - 1]:
            stt += [scannos[n]]
            stp += [dif[n]]
            rng += [1]
        else:
            rng[-1] += 1
    stt += [scannos[-1]]
    rng += [1]

    out = []
    x = 0
    while x < len(stt):
        if rng[x] == 1:
            out += ['{}'.format(stt[x])]
            x += 1
        elif stp[x] == 1:
            out += ['{}{}{}'.format(stt[x], sep, stt[x + 1])]
            x += 2
        else:
            out += ['{}{}{}{}{}'.format(stt[x], sep, stp[x], sep, stt[x + 1])]
            x += 2
    return ','.join(out)


def numbers2string(scannos, sep=':'):
    """
    Convert a list of numbers to a simple string
    E.G.
    numbers2string([50001,50002,50003]) = '5000[1:3]'
    numbers2string([51020,51030,51040]) = '510[20:10:40]'
    """

    if type(scannos) is str or type(scannos) is int or len(scannos) == 1:
        return str(scannos)

    scannos = np.sort(scannos).astype(str)

    n = len(scannos[0])
    while np.all([scannos[0][:-n] == x[:-n] for x in scannos]):
        n -= 1

    if n == len(scannos[0]):
        return '{}-{}'.format(scannos[0], scannos[-1])

    inistr = scannos[0][:-(n + 1)]
    strc = [i[-(n + 1):] for i in scannos]
    liststr = findranges(strc, sep=sep)
    return '{}[{}]'.format(inistr, liststr)


def stfm(val, err):
    """
    Create standard form string from value and uncertainty"
     str = stfm(val,err)
     Examples:
          '35.25 (1)' = stfm(35.25,0.01)
          '110 (5)' = stfm(110.25,5)
          '0.0015300 (5)' = stfm(0.00153,0.0000005)
          '1.56(2)E+6' = stfm(1.5632e6,1.53e4)

    Notes:
     - Errors less than 0.01% of values will be given as 0
     - The maximum length of string is 13 characters
     - Errors greater then 10x the value will cause the value to be rounded to zero
    """

    # Determine the number of significant figures from the error
    if err == 0. or val / float(err) >= 1E5:
        # Zero error - give value to 4 sig. fig.
        out = '{:1.5G}'.format(val)
        if 'E' in out:
            out = '{}(0)E{}'.format(*out.split('E'))
        else:
            out = out + ' (0)'
        return out
    elif np.log10(np.abs(err)) > 0.:
        # Error > 0
        sigfig = np.ceil(np.log10(np.abs(err))) - 1
        dec = 0.
    elif np.isnan(err):
        # nan error
        return '{} (-)'.format(val)
    else:
        # error < 0
        sigfig = np.floor(np.log10(np.abs(err)) + 0.025)
        dec = -sigfig

    # Round value and error to the number of significant figures
    rval = round(val / (10. ** sigfig)) * (10. ** sigfig)
    rerr = round(err / (10. ** sigfig)) * (10. ** sigfig)
    # size of value and error
    pw = np.floor(np.log10(np.abs(rval)))
    pwr = np.floor(np.log10(np.abs(rerr)))

    max_pw = max(pw, pwr)
    ln = max_pw - sigfig  # power difference

    if np.log10(np.abs(err)) < 0:
        rerr = err / (10. ** sigfig)

    # Small numbers - exponential notation
    if max_pw < -3.:
        rval = rval / (10. ** max_pw)
        fmt = '{' + '0:1.{:1.0f}f'.format(ln) + '}({1:1.0f})E{2:1.0f}'
        return fmt.format(rval, rerr, max_pw)

    # Large numbers - exponential notation
    if max_pw >= 4.:
        rval = rval / (10. ** max_pw)
        rerr = rerr / (10. ** sigfig)
        fmt = '{' + '0:1.{:1.0f}f'.format(ln) + '}({1:1.0f})E+{2:1.0f}'
        return fmt.format(rval, rerr, max_pw)

    fmt = '{' + '0:0.{:1.0f}f'.format(dec + 0) + '} ({1:1.0f})'
    return fmt.format(rval, rerr)


def value_datetime(value, date_format=None):
    """
    Convert date string or timestamp to datetime object
    :param value: str or float
    :param date_format: datetime
    :return: datetime with tzinfo removed for easy comparison
    """
    if date_format is None:
        date_format = DATE_FORMAT

    def strptime(val):
        return datetime.datetime.strptime(val, date_format)

    def timestamp(val):
        return datetime.datetime.fromtimestamp(float(val))

    timefun = [parse, strptime, timestamp]
    for fun in timefun:
        try:
            dt = fun(value)
            dt = dt.replace(tzinfo=None)  # make datetime tznieve
            return dt
        except (ValueError, TypeError):
            pass
    raise ValueError('%s cannot be converted to datetime' % value)


def data_datetime(data, date_format=None):
    """
    Convert date string to datetime object
      datetime_array = date_datetime('2020-10-22T09:33:11.894+01:00', "%Y-%m-%dT%H:%M:%S.%f%z")
     datetime_array[0] will give first time
     datetime_array[-1] will give last time
    :param data: str or list of str or list of floats
    :param date_format: str format used in datetime.strptime (see https://strftime.org/)
    :return: list of datetime
    """
    data = liststr(data)
    return np.array([value_datetime(value, date_format) for value in data])


def axes_from_cmd(cmd, alt_names=None):
    """
    Get axes name from command string
    :param cmd: str
    :param alt_names: dict {name_in_cmd: name_in_file}
    :return: str
    """
    alt_names = {} if alt_names is None else alt_names
    cmd = cmd.split()
    axes = cmd[1]
    if axes in alt_names:
        axes = alt_names[axes]
    # These are specific to I16...
    if axes == 'hkl':
        if cmd[0] == 'scan':
            hstep, kstep, lstep = cmd[8:11]
        elif cmd[0] == 'scancn':
            hstep, kstep, lstep = cmd[2:5]
        else:
            raise Warning('Warning unknown type of hkl scan')

        if float(re.sub("[^0-9.]", "", hstep)) > 0.0:
            axes = 'h'
        elif float(re.sub("[^0-9.]", "", kstep)) > 0.0:
            axes = 'k'
        else:
            axes = 'l'
    elif axes == 'energy':
        axes = 'energy2'
    elif axes == 'sr2':
        axes = 'azimuthal'  # 'phi' in pre-DiffCalc scans
    elif axes == 'th2th':
        axes = 'delta'
    elif axes == 'ppp_energy':
        axes = 'ppp_offset'
    return axes


def signal_from_cmd(cmd, alt_names=None):
    """
    Get signal name from command string
    :param cmd: str
    :param alt_names: dict {name_in_cmd: name_in_file}
    :return: str
    """
    alt_names = {} if alt_names is None else alt_names
    cmd_split = cmd.split()
    signal = 'signal'
    for signal in cmd_split[::-1]:
        try:
            float(signal)
        except ValueError:
            break
    if signal in alt_names:
        signal = alt_names[signal]
    # These are specific to I16...
    if signal == 't':
        signal = 'APD'
    elif 'roi' in signal:
        signal = signal + '_sum'
    elif 'pil100k' in cmd:
        signal = 'sum'
    elif 'pil3_100k' in cmd:
        signal = 'sum'
    elif 'pil2m' in cmd:
        signal = 'sum'
    elif 'merlin' in cmd:
        signal = 'sum'
    elif 'bpm' in cmd:
        signal = 'sum'
    elif 'QBPM6' in cmd:
        signal = 'C1'
    return signal


def axis_repeat(*args):
    """
    Determine the repeating pattern in a or several axes
    :param args: [n,] length array
    :return: repeat_length
    """

    rep_len = []
    for arg in args:
        delta = np.abs(np.diff(arg))
        ch_idx = np.append(-1, np.where(delta > delta.max() * 0.9))  # find biggest changes
        ch_delta = np.diff(ch_idx)
        rep_len += [np.round(np.mean(ch_delta))]
    return int(max(rep_len))


def square_array(xaxis, yaxis, zaxis=None, repeat_length=None):
    """
    Reshape 1D x/y axes into 2D axes, estimating the repeat length
    :param xaxis: [n,] list or array
    :param yaxis: [n,] list or array
    :param zaxis: [n,] list or array, or None
    :param repeat_length: int m, value to reshape array (None to determine automatically)
    :return: xaxis, yaxis, [zaxis] [n//m, m] array
    """
    xaxis = np.asarray(xaxis).reshape(-1)
    yaxis = np.asarray(yaxis).reshape(-1)

    # Determine the repeat length of the scans
    if repeat_length is None:
        repeat_length = axis_repeat(xaxis, yaxis)

    # Reshape into square arrays
    # If this is problematic, look at scipy.interpolate.griddata
    xaxis = xaxis[:repeat_length * (len(xaxis) // repeat_length)].reshape(-1, repeat_length)
    yaxis = yaxis[:repeat_length * (len(yaxis) // repeat_length)].reshape(-1, repeat_length)
    if zaxis is not None:
        zaxis = np.asarray(zaxis).reshape(-1)
        zaxis = zaxis[:repeat_length * (len(zaxis) // repeat_length)].reshape(-1, repeat_length)
        return xaxis, yaxis, zaxis
    return xaxis, yaxis


def time_difference(start_time, end_time=None):
    """
    Return time difference between first and last time
    :param start_time: str or list
    :param end_time: None or str
    :return: datetime.timedelta
    """
    start_time = data_datetime(start_time)[0]
    if end_time is None:
        end_time = data_datetime(start_time)[-1]
    else:
        end_time = data_datetime(end_time)[0]
    time_delta = end_time - start_time
    return time_delta


def check_naughty_eval(eval_str):
    """
    Check str for naughty eval arguments such as os or import
    This is not foolproof.
    :param eval_str: str
    :return: pass or raise error
    """
    from .__settings__ import EVAL_MODE
    if not EVAL_MODE:
        raise Exception('EVAL_MODE is not active')
    bad_names = ['import', 'os.', 'sys.']
    for bad in bad_names:
        if bad in eval_str:
            raise Exception('This operation is not allowed as it contains: "%s"' % bad)


def function_generator(operation):
    """
    Generate a function from an operation on "x"
      fn = function_generator("np.sqrt(x + 0.1)")
    :param operation: str operation acting on variable "x", or function
    :return: function
    """
    if hasattr(operation, '__call__'):
        return operation
    check_naughty_eval(operation)
    function_str = "lambda x: %s" % operation
    return eval(function_str)


def load_from_config(config_file):
    """
    Load config settings from instrument.config file.
      .config files should be json files with the following keys:
        'name': str
        'default_names': dict,
        'formats': dict,
        'default_values': dict,
        'options': dict
    :param config_file: str config filename
    :return: name, default_names, formats, default_values, options
    """
    if os.path.isfile(os.path.join(CONFIG_DIR, config_file)):
        config_file = os.path.join(CONFIG_DIR, config_file)
    elif os.path.isfile(os.path.join(CONFIG_DIR, config_file + '.config')):
        config_file = os.path.join(CONFIG_DIR, config_file + '.config')
    with open(config_file, 'r') as fp:
        config = json.load(fp)
    name = config['name'] if 'name' in config else 'None'
    default_names = config['default_names'] if 'default_names' in config else {}
    formats = config['formats'] if 'formats' in config else {}
    default_values = config['default_values'] if 'default_values' in config else {}
    options = config['options'] if 'options' in config else {}
    options['config_file'] = config_file
    return name, default_names, formats, default_values, options


def save_to_config(config_file=None, name='None', default_names=None, formats=None, default_values=None, options=None):
    """
    Saves config settings to instrument.config file.
      .config files should be json files with the following keys:
        'name': str
        'default_names': dict,
        'formats': dict,
        'default_values': dict,
        'options': dict
    :param config_file: str config filename
    :param name: str
    :param default_names: dict
    :param formats: dict
    :param default_values: dict
    :param options: dict
    :return: None
    """
    if config_file is None:
        if options is not None and 'config_file' in options:
            config_file = options['config_file']
        else:
            raise IOError('config_file not found in options')

    config = {
        'name': name, 'default_names': {} if default_names is None else default_names,
        'formats': {} if formats is None else formats,
        'default_values': {} if default_values is None else default_values,
        'options': {} if options is None else options
    }

    with open(config_file, 'w') as fp:
        json.dump(config, fp, sort_keys=True, indent=4)
    print('config file written: %s' % config_file)
