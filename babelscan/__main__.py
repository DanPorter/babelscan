"""
BabelScan
A generic class for reading many types of scan data that you don't need to put in your ear.

Usage:
  $ python -i -m babelscan
  > start interactive terminal with various modules loaded
OR
  $ python -m bablescan /some/scan/file.nxs
   - displays information in the file

By Dan Porter, PhD
Diamond Light Source Ltd.
2021
"""
if __name__ == '__main__':

    import sys
    import numpy as np
    import h5py
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('..matplotlib not available..')
    import babelscan
    from babelscan import file_loader, load_files, FolderMonitor, find_files
    from babelscan import load_hdf_values, hdf_loader
    from babelscan import Instrument, instrument_from_config, save_to_config

    print('\nBabelScan version %s, %s' % (babelscan.__version__, babelscan.__date__))
    print(' By Dan Porter, Diamond Light Source Ltd.')
    print('See help(babelscan) or start with: scan = file_loader("/some/scan/file.nxs")')

    for arg in sys.argv:
        if '.py' in arg:
            continue
        try:
            print('Opening: %s' % arg)
            scan = file_loader(arg)
            print(scan)

            if hasattr(scan, 'tree'):
                print('\n\nFile data: %s:\n%s' % (arg, scan.tree(detail=True)))
        except Exception:
            pass

