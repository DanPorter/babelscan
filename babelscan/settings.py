"""
babelscan configuration settings
"""

from .__settings__ import EVAL_MODE, PLOTTING_MODE


def init_scan_plot_manager(scan):
    """Initialise plotting"""
    if PLOTTING_MODE.lower() in ['matplotlib', 'pyplot']:
        try:
            from .plotting_matplotlib import ScanPlotManager
            manager = ScanPlotManager(scan)
        except ImportError:
            manager = None
    else:
        manager = None
    return manager


def init_multiscan_plot_manager(multiscan):
    """Initialise plotting"""
    if PLOTTING_MODE.lower() in ['matplotlib', 'pyplot']:
        try:
            from .plotting_matplotlib import MultiScanPlotManager
            manager = MultiScanPlotManager(multiscan)
        except ImportError:
            manager = None
    else:
        manager = None
    return manager


def init_scan_fit_manager(scan):
    """Initialise plotting"""
    try:
        from .fitting import ScanFitManager
        manager = ScanFitManager(scan)
    except ImportError:
        manager = None
    return manager


def init_multiscan_fit_manager(multiscan):
    """Initialise plotting"""
    try:
        from .fitting import MultiScanFitManager
        manager = MultiScanFitManager(multiscan)
    except ImportError:
        manager = None
    return manager


def init_vol_plot_manager(vol):
    """Initialise plotting"""
    if PLOTTING_MODE.lower() in ['matplotlib', 'pyplot']:
        try:
            from .plotting_matplotlib import VolumePlotManager
            manager = VolumePlotManager(vol)
        except ImportError:
            manager = None
    else:
        manager = None
    return manager



