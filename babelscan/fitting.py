"""
Fitting functions using lmfit

See: https://lmfit.github.io/lmfit-py/builtin_models.html

Use of peakfit:
from fitting import peakfit
fit = peakfit(xdata, ydata)  # returns lmfit object
print(fit)
fit.plot()
"""

import numpy as np
from lmfit.models import GaussianModel, LorentzianModel, VoigtModel, PseudoVoigtModel, LinearModel, ExponentialModel

# https://lmfit.github.io/lmfit-py/builtin_models.html#peak-like-models
MODELS = {
    'gaussian': GaussianModel,
    'lorentz': LorentzianModel,
    'voight': VoigtModel,
    'pvoight': PseudoVoigtModel,
    'linear': LinearModel,
    'exponential': ExponentialModel
}

PEAK_MODELS = {
    'gaussian': ['gaussian', 'gauss'],
    'voight': ['voight', 'voight model'],
    'pvoight': ['pseudovoight', 'pvoight'],
    'lorentz': ['lorentz', 'lorentzian', 'lor'],
}

BACKGROUND_MODELS = {
    'linear': ['flat', 'slope', 'linear', 'line', 'straight'],
    'exponential': ['exponential', 'curve']
}

# https://lmfit.github.io/lmfit-py/fitting.html#fit-methods-table
METHODS = {
    'leastsq': 'Levenberg-Marquardt',
    'nelder': 'Nelder-Mead',
    'lbfgsb': 'L-BFGS-B',
    'powell': 'Powell',
    'cg': 'Conjugate Gradient',
    'newton': 'Newton-CG',
    'cobyla': 'COBYLA',
    'bfgsb': 'BFGS',
    'tnc': 'Truncated Newton',
    'trust-ncg': 'Newton CG trust-region',
    'trust-exact': 'Exact trust-region',
    'trust-krylov': 'Newton GLTR trust-region',
    'trust-constr': 'Constrained trust-region',
    'dogleg': 'Dogleg',
    'slsqp': 'Sequential Linear Squares Programming',
    'differential_evolution': 'Differential Evolution',
    'brute': 'Brute force method',
    'basinhopping': 'Basinhopping',
    'ampgo': 'Adaptive Memory Programming for Global Optimization',
    'shgo': 'Simplicial Homology Global Ooptimization',
    'dual_annealing': 'Dual Annealing',
    'emcee': 'Maximum likelihood via Monte-Carlo Markov Chain',
}


def error_func(y):
    """Default error function"""
    return np.sqrt(np.abs(y) + 1)


def peak_ratio(y, yerror=None):
    """
    Return the ratio signal / error for given dataset
    From Blessing, J. Appl. Cryst. (1997). 30, 421-426 Equ: (1) + (6)
      peak_ratio = (sum((y-bkg)/dy^2)/sum(1/dy^2)) / sqrt(i/sum(1/dy^2))
    :param y: array of y data
    :param yerror: array of errors on data, or None to calcualte np.sqrt(y+0.001)
    :return: float ratio signal / err
    """
    if yerror is None:
        yerror = error_func(y)
    bkg = np.min(y)
    wi = 1 / yerror ** 2
    signal = np.sum(wi * (y - bkg)) / np.sum(wi)
    err = np.sqrt(len(y) / np.sum(wi))
    return signal / err


def gen_weights(yerrors=None):
    """
    Generate weights for fitting routines
    :param yerrors: array(n) or None
    :return: array(n) or None
    """
    if yerrors is None or np.all(np.abs(yerrors) < 0.001):
        weights = None
    else:
        weights = 1 / np.array(yerrors, dtype=float)
        weights = np.abs(np.nan_to_num(weights))
        weights[weights < 1] = 1.0
    return weights


def gauss(x, y=None, height=1, cen=0, fwhm=0.5, bkg=0):
    """
    Define Gaussian distribution in 1 or 2 dimensions
    From http://fityk.nieto.pl/model.html
        x = [1xn] array of values, defines size of gaussian in dimension 1
        y = None* or [1xm] array of values, defines size of gaussian in dimension 2
        height = peak height
        cen = peak centre
        fwhm = peak full width at half-max
        bkg = background
    """

    if y is None:
        y = cen

    x = np.asarray(x, dtype=np.float).reshape([-1])
    y = np.asarray(y, dtype=np.float).reshape([-1])
    X, Y = np.meshgrid(x, y)
    g = height * np.exp(-np.log(2) * (((X - cen) ** 2 + (Y - cen) ** 2) / (fwhm / 2) ** 2)) + bkg

    if len(y) == 1:
        g = g.reshape([-1])
    return g


def group_adjacent(values, close=10):
    """
    Average adjacent values in array, return grouped array and indexes to return groups to original array
    E.G.
     grp, idx = group_adjacent([1,2,3,10,12,31], close=3)
     grp -> [2, 11, 31]
     idx -> [[0,1,2], [3,4], [5]]

    :param values: array of values to be grouped
    :param close: float
    :return grouped_values: float array(n) of grouped values
    :return indexes: [n] list of lists, each item relates to an averaged group, with indexes from values
    """
    # Check distance between good peaks
    dist_chk = []
    dist_idx = []
    gx = 0
    dist = [values[gx]]
    idx = [gx]
    while gx < len(values) - 1:
        gx += 1
        if (values[gx] - values[gx - 1]) < close:
            dist += [values[gx]]
            idx += [gx]
            # print('Close %2d %2d %2d  %s' % (gx, indexes[gx], indexes[gx-1], dist))
        else:
            dist_chk += [np.mean(dist)]
            dist_idx += [idx]
            dist = [values[gx]]
            idx = [gx]
            # print('Next %2d %2d %2d %s' % (gx, indexes[gx], indexes[gx-1], dist_chk))
    dist_chk += [np.mean(dist)]
    dist_idx += [idx]
    # print('Last %2d %2d %2d %s' % (gx, indexes[gx], indexes[gx-1], dist_chk))
    return np.array(dist_chk), dist_idx


def local_maxima_1d(y):
    """
    Find local maxima in 1d array
    Returns points with central point higher than neighboring points
    Copied from scipy.signal._peak_finding_utils
    https://github.com/scipy/scipy/blob/v1.7.1/scipy/signal/_peak_finding_utils.pyx
    :param y: list or array
    :return: array of peak indexes
    """
    y = np.asarray(y, dtype=float).reshape(-1)

    # Preallocate, there can't be more maxima than half the size of `y`
    midpoints = np.empty(y.shape[0] // 2, dtype=np.intp)
    m = 0  # Pointer to the end of valid area in allocated arrays
    i = 1  # Pointer to current sample, first one can't be maxima
    i_max = y.shape[0] - 1  # Last sample can't be maxima
    while i < i_max:
        # Test if previous sample is smaller
        if y[i - 1] < y[i]:
            i_ahead = i + 1  # Index to look ahead of current sample

            # Find next sample that is unequal to x[i]
            while i_ahead < i_max and y[i_ahead] == y[i]:
                i_ahead += 1

            # Maxima is found if next unequal sample is smaller than x[i]
            if y[i_ahead] < y[i]:
                left_edge = i
                right_edge = i_ahead - 1
                midpoints[m] = (left_edge + right_edge) // 2
                m += 1
                # Skip samples that can't be maximum
                i = i_ahead
        i += 1
    return midpoints[:m]


def find_local_maxima(y, yerror=None):
    """
    Find local maxima in 1d arrays, returns index of local maximums, plus
    estimation of the peak power for each maxima and a classification of whether the maxima is greater than
    the standard deviation of the error
    E.G.
        index, power, isgood = find_local_maxima(ydata)
        maxima = ydata[index[isgood]]
        maxima_power = power[isgood]
    Peak Power:
      peak power for each maxima is calculated using the peak_ratio algorithm for each maxima and adjacent points
    Good Peaks:
      Maxima are returned Good if:  power > (max(y) - min(y)) / std(yerror)
    :param y: array(n) of data
    :param yerror: array(n) of errors on data, or None to use default error function (sqrt(abs(y)+1))
    :return index: array(m<n) of indexes in y of maxima
    :return power: array(m) of estimated peak power for each maxima
    :return isgood: bool array(m) where True elements have power > power of the array
    """

    if yerror is None or np.all(np.abs(yerror) < 0.1):
        yerror = error_func(y)
    yerror[yerror < 1] = 1.0
    bkg = np.min(y)
    wi = 1 / yerror ** 2

    index = local_maxima_1d(y)
    # average nearest 3 points to peak
    power = np.array([np.sum(wi[m-1:m+2] * (y[m-1:m+2] - bkg)) / np.sum(wi[m-1:m+2]) for m in index])
    # Determine if peak is good
    isgood = power > (np.max(y) - np.min(y)) / (np.std(yerror) + 1)
    return index, power, isgood


def find_peaks(y, yerror=None, min_peak_power=None, peak_distance_idx=6):
    """
    Find peak shaps in linear-spaced 1d arrays with poisson like numerical values
    E.G.
      index, power = find_peaks(ydata, yerror, min_peak_power=None, peak_distance_idx=10)
      peak_centres = xdata[index]  # ordered by peak strength
    :param y: array(n) of data
    :param yerror: array(n) of errors on data, or None to use default error function (sqrt(abs(y)+1))
    :param min_peak_power: float, only return peaks with power greater than this. If None compare against std(y)
    :param peak_distance_idx: int, group adjacent maxima if closer in index than this
    :return index: array(m) of indexes in y of peaks that satisfy conditions
    :return power: array(m) of estimated power of each peak
    """
    # Get all peak positions
    midpoints, peak_signals, chk = find_local_maxima(y, yerror)

    if min_peak_power is None:
        good_peaks = chk
    else:
        good_peaks = peak_signals >= min_peak_power

    # select indexes of good peaks
    peaks_idx = midpoints[good_peaks]
    peak_power = peak_signals[good_peaks]
    if len(peaks_idx) == 0:
        return peaks_idx, peak_power

    # Average peaks close to each other
    group_idx, group_signal_idx = group_adjacent(peaks_idx, peak_distance_idx)
    peaks_idx = np.round(group_idx).astype(int)
    peak_power = np.array([np.sum(peak_power[ii]) for ii in group_signal_idx])

    # sort peak order by strength
    power_sort = np.argsort(peak_power)
    return peaks_idx[power_sort], peak_power[power_sort]


def modelfit(xvals, yvals, yerrors=None, model=None, initial_parameters=None, fix_parameters=None,
             method='leastsq', print_result=False, plot_result=False):
    """
    Fit x,y data to a model from lmfit
    E.G.:
      res = peakfit(x, y, model='Gauss')
      print(res.fit_report())
      res.plot()
      val = res.params['amplitude'].value
      err = res.params['amplitude'].stderr

    Model:
     from lmfit import models
     model1 = model.GaussianModel()
     model2 = model.LinearModel()
     model = model1 + model2
     res = model.fit(y, x=x)

    Provide initial guess:
      res = modelfit(x, y, model=VoightModel(), initial_parameters={'center':1.23})

    Fix parameter:
      res = modelfit(x, y, model=VoightModel(), fix_parameters={'sigma': fwhm/2.3548200})

    :param xvals: array(n) position data
    :param yvals: array(n) intensity data
    :param yerrors: None or array(n) - error data to pass to fitting function as weights: 1/errors^2
    :param model: lmfit.Model
    :param initial_parameters: None or dict of initial values for parameters
    :param fix_parameters: None or dict of parameters to fix at positions
    :param method: str method name, from lmfit fitting methods
    :param print_result: if True, prints the fit results using fit.fit_report()
    :param plot_result: if True, plots the results using fit.plot()
    :return: lmfit.model.ModelResult < fit results object
    """

    xvals = np.asarray(xvals, dtype=float).reshape(-1)
    yvals = np.asarray(yvals, dtype=float).reshape(-1)
    weights = gen_weights(yerrors)

    if initial_parameters is None:
        initial_parameters = {}
    if fix_parameters is None:
        fix_parameters = {}

    if model is None:
        model = GaussianModel() + LinearModel()

    pars = model.make_params()

    # user input parameters
    for ipar, ival in initial_parameters.items():
        if ipar in pars:
            pars[ipar].set(value=ival, vary=True)
    for ipar, ival in fix_parameters.items():
        if ipar in pars:
            pars[ipar].set(value=ival, vary=False)

    res = model.fit(yvals, pars, x=xvals, weights=weights, method=method)

    if print_result:
        print(res.fit_report())
    if plot_result:
        res.plot()
    return res


def peakfit(xvals, yvals, yerrors=None, model='Voight', background='slope',
            initial_parameters=None, fix_parameters=None, method='leastsq', print_result=False, plot_result=False):
    """
    Fit x,y data to a peak model using lmfit
    E.G.:
      res = peakfit(x, y, model='Gauss')
      print(res.fit_report())
      res.plot()
      val = res.params['amplitude'].value
      err = res.params['amplitude'].stderr

    Peak Models:
     Choice of peak model: 'Gaussian', 'Lorentzian', 'Voight',' PseudoVoight'
    Background Models:
     Choice of background model: 'slope', 'exponential'

    Peak Parameters:
     'amplitude', 'center', 'sigma', pvoight only: 'fraction'
     output only: 'fwhm', 'height'
    Background parameters:
     'bkg_slope', 'bkg_intercept', or for exponential: 'bkg_amplitude', 'bkg_decay'

    Provide initial guess:
      res = peakfit(x, y, model='Voight', initial_parameters={'center':1.23})

    Fix parameter:
      res = peakfit(x, y, model='gauss', fix_parameters={'sigma': fwhm/2.3548200})

    :param xvals: array(n) position data
    :param yvals: array(n) intensity data
    :param yerrors: None or array(n) - error data to pass to fitting function as weights: 1/errors^2
    :param model: str, specify the peak model: 'Gaussian','Lorentzian','Voight'
    :param background: str, specify the background model: 'slope', 'exponential'
    :param initial_parameters: None or dict of initial values for parameters
    :param fix_parameters: None or dict of parameters to fix at positions
    :param method: str method name, from lmfit fitting methods
    :param print_result: if True, prints the fit results using fit.fit_report()
    :param plot_result: if True, plots the results using fit.plot()
    :return: lmfit.model.ModelResult < fit results object
    """

    xvals = np.asarray(xvals, dtype=float).reshape(-1)
    yvals = np.asarray(yvals, dtype=float).reshape(-1)
    weights = gen_weights(yerrors)

    if initial_parameters is None:
        initial_parameters = {}
    if fix_parameters is None:
        fix_parameters = {}

    peak_mod = None
    bkg_mod = None
    for model_name, names in PEAK_MODELS.items():
        if model.lower() in names:
            peak_mod = MODELS[model_name]()
    for model_name, names in BACKGROUND_MODELS.items():
        if background.lower() in names:
            bkg_mod = MODELS[model_name](prefix='bkg_')

    pars = peak_mod.guess(yvals, x=xvals)
    pars += bkg_mod.make_params()
    # pars += bkg_mod.make_params(intercept=np.min(yvals), slope=0)
    # pars['gamma'].set(value=0.7, vary=True, expr='') # don't fix gamma

    # user input parameters
    for ipar, ival in initial_parameters.items():
        if ipar in pars:
            pars[ipar].set(value=ival, vary=True)
    for ipar, ival in fix_parameters.items():
        if ipar in pars:
            pars[ipar].set(value=ival, vary=False)

    mod = peak_mod + bkg_mod
    res = mod.fit(yvals, pars, x=xvals, weights=weights, method=method)

    if print_result:
        print(res.fit_report())
    if plot_result:
        res.plot()
    return res


def peak2dfit(xdata, ydata, image_data, initial_parameters=None, fix_parameters=None,
              print_result=False, plot_result=False):
    """
    Fit Gaussian Peak in 2D
    *** requires lmfit > 1.0.3 ***
        Not yet finished!
    :param xdata:
    :param ydata:
    :param image_data:
    :param initial_parameters:
    :param fix_parameters:
    :param print_result:
    :param plot_result:
    :return:
    """
    from lmfit.models import Gaussian2dModel  # lmfit V1.0.3+
    print('Not yet finished...')
    pass


def multipeakfit(xvals, yvals, yerrors=None,
                 npeaks=None, min_peak_power=None, peak_distance_idx=10,
                 model='Gaussian', background='slope', initial_parameters=None, fix_parameters=None, method='leastsq',
                 print_result=False, plot_result=False):
    """
    Fit x,y data to a model with multiple peaks using lmfit
    See: https://lmfit.github.io/lmfit-py/builtin_models.html#example-3-fitting-multiple-peaks-and-using-prefixes
    E.G.:
      res = peakfit(x, y, model='Gauss')
      print(res.fit_report())
      res.plot()
      val = res.params['amplitude'].value
      err = res.params['amplitude'].stderr

    Peak Search:
     The number of peaks and initial peak centers will be estimated using the find_peaks function. If npeaks is given,
     the largest npeaks will be used initially. 'min_peak_power' and 'peak_distance_idx' can be input to tailor the
     peak search results.
     If the peak search returns < npeaks, fitting parameters will initially choose npeaks equally distributed points

    Peak Models:
     Choice of peak model: 'Gaussian', 'Lorentzian', 'Voight',' PseudoVoight'
    Background Models:
     Choice of background model: 'slope', 'exponential'

    Peak Parameters (%d=number of peak):
    Parameters in '.._parameters' dicts and in output results. Each peak (upto npeaks) has a set number of parameters:
     'p%d_amplitude', 'p%d_center', 'p%d_dsigma', pvoight only: 'p%d_fraction'
     output only: 'p%d_fwhm', 'p%d_height'
    Background parameters:
     'bkg_slope', 'bkg_intercept', or for exponential: 'bkg_amplitude', 'bkg_decay'

    Provide initial guess:
      res = multipeakfit(x, y, model='Voight', initial_parameters={'p1_center':1.23})

    Fix parameter:
      res = multipeakfit(x, y, model='gauss', fix_parameters={'p1_sigma': fwhm/2.3548200})

    :param xvals: array(n) position data
    :param yvals: array(n) intensity data
    :param yerrors: None or array(n) - error data to pass to fitting function as weights: 1/errors^2
    :param npeaks: None or int number of peaks to fit. None will guess the number of peaks
    :param min_peak_power: float, only return peaks with power greater than this. If None compare against std(y)
    :param peak_distance_idx: int, group adjacent maxima if closer in index than this
    :param model: str or lmfit.Model, specify the peak model 'Gaussian','Lorentzian','Voight'
    :param background: str, specify the background model: 'slope', 'exponential'
    :param initial_parameters: None or dict of initial values for parameters
    :param fix_parameters: None or dict of parameters to fix at positions
    :param method: str method name, from lmfit fitting methods
    :param print_result: if True, prints the fit results using fit.fit_report()
    :param plot_result: if True, plots the results using fit.plot()
    :return: lmfit.model.ModelResult < fit results object
    """
    xvals = np.asarray(xvals, dtype=float).reshape(-1)
    yvals = np.asarray(yvals, dtype=float).reshape(-1)
    weights = gen_weights(yerrors)

    # Find peaks
    peak_idx, peak_pow = find_peaks(yvals, yerrors, min_peak_power, peak_distance_idx)
    peak_centers = {'p%d_center' % (n+1): xvals[peak_idx[n]] for n in range(len(peak_idx))}
    if npeaks is None:
        npeaks = len(peak_centers)
        print('Fitting %d peaks:' % npeaks)

    if initial_parameters is None:
        initial_parameters = {}
    if fix_parameters is None:
        fix_parameters = {}

    peak_mod = None
    bkg_mod = None
    for model_name, names in PEAK_MODELS.items():
        if model.lower() in names:
            peak_mod = MODELS[model_name]
    for model_name, names in BACKGROUND_MODELS.items():
        if background.lower() in names:
            bkg_mod = MODELS[model_name]

    mod = bkg_mod(prefix='bkg_')
    for n in range(npeaks):
        mod += peak_mod(prefix='p%d_' % (n+1))

    pars = mod.make_params()

    # initial parameters
    min_wid = np.mean(np.diff(xvals))
    max_wid = xvals.max() - xvals.min()
    area = (yvals.max() - yvals.min()) * (3 * min_wid)
    percentile = np.linspace(0, 100, npeaks + 2)
    for n in range(1, npeaks+1):
        pars['p%d_amplitude' % n].set(value=area/npeaks, min=0)
        pars['p%d_sigma' % n].set(value=3*min_wid, min=min_wid, max=max_wid)
        pars['p%d_center' % n].set(value=np.percentile(xvals, percentile[n]), min=xvals.min(), max=xvals.max())
    # find_peak centers
    for ipar, ival in peak_centers.items():
        if ipar in pars:
            pars[ipar].set(value=ival, vary=True)
    # user input parameters
    for ipar, ival in initial_parameters.items():
        if ipar in pars:
            pars[ipar].set(value=ival, vary=True)
    for ipar, ival in fix_parameters.items():
        if ipar in pars:
            pars[ipar].set(value=ival, vary=False)

    # Fit data against model using choosen method
    res = mod.fit(yvals, pars, x=xvals, weights=weights, method=method)

    if print_result:
        print(res.fit_report())
    if plot_result:
        fig, grid = res.plot()
        ax1, ax2 = fig.axes
        # Add peak components
        comps = res.eval_components(x=xvals)
        for component in comps.keys():
            ax2.plot(xvals, comps[component], label=component)
            ax2.legend()
    return res


"----------------------------------------------------------------------------------------------------------------------"
"------------------------------------------------ ScanFitManager ------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"


class ScanFitManager:
    """
    ScanFitManager
     Holds several functions for automatically fitting scan data

    :param scan: babelscan.Scan
    """

    def __init__(self, scan):
        self.scan = scan

    def __call__(self, *args, **kwargs):
        """Calls ScanFitManager.fit(...)"""
        return self.fit(*args, **kwargs)

    def __str__(self):
        return self.fit_report()

    def peak_ratio(self, yaxis='signal'):
        """
        Return the ratio signal / error for given dataset
        From Blessing, J. Appl. Cryst. (1997). 30, 421-426 Equ: (1) + (6)
          peak_ratio = (sum((y-bkg)/dy^2)/sum(1/dy^2)) / sqrt(i/sum(1/dy^2))
        :param yaxis: str name or address of array to plot on y axis
        :return: float ratio signal / err
        """
        xdata, ydata, yerror, xname, yname = self.scan.get_plot_data('axes', yaxis, None, None)
        return peak_ratio(ydata, yerror)

    def find_peaks(self, xaxis='axes', yaxis='signal', min_peak_power=None, peak_distance_idx=10):
        """
        Find peak shaps in linear-spaced 1d arrays with poisson like numerical values
        E.G.
          centres, index, power = self.find_peaks(xaxis, yaxis, min_peak_power=None, peak_distance_idx=10)
        :param xaxis: str name or address of array to plot on x axis
        :param yaxis: str name or address of array to plot on y axis
        :param min_peak_power: float, only return peaks with power greater than this. If None compare against std(y)
        :param peak_distance_idx: int, group adjacent maxima if closer in index than this
        :return centres: array(m) of peak centers in x, equiv. to xdata[index]
        :return index: array(m) of indexes in y of peaks that satisfy conditions
        :return power: array(m) of estimated power of each peak
        """
        xdata, ydata, yerror, xname, yname = self.scan.get_plot_data(xaxis, yaxis, None, None)
        index, power = find_peaks(ydata, yerror, min_peak_power, peak_distance_idx)
        return xdata[index], index, power

    def fit(self, xaxis='axes', yaxis='signal', model='Gaussian', background='slope',
            initial_parameters=None, fix_parameters=None, method='leastsq', print_result=False, plot_result=False):
        """
        Fit x,y data to a peak model using lmfit
        E.G.:
          res = self.fit('axes', 'signal', model='Gauss')
          print(res.fit_report())
          res.plot()
          val = res.params['amplitude'].value
          err = res.params['amplitude'].stderr

        Peak Models:
         Choice of peak model: 'Gaussian', 'Lorentzian', 'Voight',' PseudoVoight'
        Background Models:
         Choice of background model: 'slope', 'exponential'

        Peak Parameters (%d=number of peak):
         'amplitude', 'center', 'sigma', pvoight only: 'fraction'
         output only: 'fwhm', 'height'
        Background parameters:
         'bkg_slope', 'bkg_intercept', or for exponential: 'bkg_amplitude', 'bkg_decay'

        Provide initial guess:
          res = self.fit(x, y, model='Voight', initial_parameters={'p1_center':1.23})

        Fix parameter:
          res = self.fit(x, y, model='gauss', fix_parameters={'p1_sigma': fwhm/2.3548200})

        :param xaxis: str name or address of array to plot on x axis
        :param yaxis: str name or address of array to plot on y axis
        :param model: str, specify the peak model 'Gaussian','Lorentzian','Voight'
        :param background: str, specify the background model: 'slope', 'exponential'
        :param initial_parameters: None or dict of initial values for parameters
        :param fix_parameters: None or dict of parameters to fix at positions
        :param method: str method name, from lmfit fitting methods
        :param print_result: if True, prints the fit results using fit.fit_report()
        :param plot_result: if True, plots the results using fit.plot()
        :return: lmfit.model.ModelResult < fit results object
        """
        xdata, ydata, yerror, xname, yname = self.scan.get_plot_data(xaxis, yaxis, None, None)

        # lmfit
        out = peakfit(xdata, ydata, yerror, model=model, background=background,
                      initial_parameters=initial_parameters, fix_parameters=fix_parameters, method=method)

        self.scan.add2namespace('lmfit', out, 'fit_result')
        fit_dict = {}
        for pname, param in out.params.items():
            ename = 'stderr_' + pname
            fit_dict[pname] = param.value
            fit_dict[ename] = param.stderr
        for name, value in fit_dict.items():
            self.scan.add2namespace(name, value)
        self.scan.add2namespace('fit', out.best_fit, other_names=['fit_%s' % yname])
        comps = out.eval_components(x=xdata)
        for component in comps:
            self.scan.add2namespace('%sfit' % component, comps[component])
        # Background
        self.scan.add2namespace('background', np.mean(comps['bkg_']))

        if print_result:
            print(self.scan.title())
            print(out.fit_report())
        if plot_result:
            fig, grid = out.plot()
            fig.suptitle(self.scan.title(), fontsize=12)
            # plt.subplots_adjust(top=0.85, left=0.15)
            ax1, ax2 = fig.axes
            ax1.set_title('')
            ax2.set_xlabel(xname)
            ax2.set_ylabel(yname)
        return out

    def multi_peak_fit(self, xaxis='axes', yaxis='signal',
                       npeaks=None, min_peak_power=None, peak_distance_idx=10,
                       model='Gaussian', background='slope',
                       initial_parameters=None, fix_parameters=None, method='leastsq',
                       print_result=False, plot_result=False):
        """
        Fit x,y data to a peak model using lmfit
        E.G.:
          res = self.multi_peak_fit('axes', 'signal', npeaks=2, model='Gauss')
          print(res.fit_report())
          res.plot()
          val1 = res.params['p1_amplitude'].value
          val2 = res.params['p2_amplitude'].value

        Peak centers:
         Will attempt a fit using 'npeaks' peaks, with centers defined by defalult by the find_peaks function
          if 'npeaks' is None, the number of peaks found by find_peaks will determine npeaks
          if 'npeaks' is greater than the number of peaks found by find_peaks, initial peak centers are evenly
          distrubuted along xdata.

        Peak Models:
         Choice of peak model: 'Gaussian', 'Lorentzian', 'Voight',' PseudoVoight'
        Background Models:
         Choice of background model: 'slope', 'exponential'

        Peak Parameters (%d=number of peak):
         'p%d_amplitude', 'p%d_center', 'p%d_sigma', pvoight only: 'p%d_fraction'
         output only: 'p%d_fwhm', 'p%d_height'
        Background parameters:
         'bkg_slope', 'bkg_intercept', or for exponential: 'bkg_amplitude', 'bkg_decay'
        Total parameters (always available, output only - sum/averages of all peaks):
         'amplitude', 'center', 'sigma', 'fwhm', 'height', 'background'

        Provide initial guess:
          res = self.multi_peak_fit(x, y, model='Voight', initial_parameters={'p1_center':1.23})

        Fix parameter:
          res = self.multi_peak_fit(x, y, model='gauss', fix_parameters={'p1_sigma': fwhm/2.3548200})

        :param xaxis: str name or address of array to plot on x axis
        :param yaxis: str name or address of array to plot on y axis
        :param npeaks: None or int number of peaks to fit. None will guess the number of peaks
        :param min_peak_power: float, only return peaks with power greater than this. If None compare against std(y)
        :param peak_distance_idx: int, group adjacent maxima if closer in index than this
        :param model: str, specify the peak model 'Gaussian','Lorentzian','Voight'
        :param background: str, specify the background model: 'slope', 'exponential'
        :param initial_parameters: None or dict of initial values for parameters
        :param fix_parameters: None or dict of parameters to fix at positions
        :param method: str method name, from lmfit fitting methods
        :param print_result: if True, prints the fit results using fit.fit_report()
        :param plot_result: if True, plots the results using fit.plot()
        :return: lmfit.model.ModelResult < fit results object
        """
        xdata, ydata, yerror, xname, yname = self.scan.get_plot_data(xaxis, yaxis, None, None)

        # lmfit
        out = multipeakfit(xdata, ydata, yerror, npeaks=npeaks, min_peak_power=min_peak_power,
                           peak_distance_idx=peak_distance_idx, model=model, background=background,
                           initial_parameters=initial_parameters, fix_parameters=fix_parameters, method=method)

        self.scan.add2namespace('lmfit', out, 'fit_result')
        fit_dict = {}
        for pname, param in out.params.items():
            ename = 'stderr_' + pname
            fit_dict[pname] = param.value
            fit_dict[ename] = param.stderr
        for name, value in fit_dict.items():
            self.scan.add2namespace(name, value)
        self.scan.add2namespace('fit', out.best_fit, other_names=['fit_%s' % yname])
        # Add peak components
        comps = out.eval_components(x=xdata)
        for component in comps:
            self.scan.add2namespace('%sfit' % component, comps[component])
        # Totals
        peak_prefx = [mod.prefix for mod in out.components if 'bkg' not in mod.prefix]
        self.scan.add2namespace('npeaks', len(peak_prefx))
        nn = 1/len(peak_prefx) if len(peak_prefx) > 0 else 1
        totals = {
            'amplitude': np.sum([out.params['%samplitude' % pfx].value for pfx in peak_prefx]),
            'center': np.mean([out.params['%scenter' % pfx].value for pfx in peak_prefx]),
            'sigma': np.mean([out.params['%ssigma' % pfx].value for pfx in peak_prefx]),
            'height': np.mean([out.params['%sheight' % pfx].value for pfx in peak_prefx]),
            'fwhm': np.mean([out.params['%sfwhm' % pfx].value for pfx in peak_prefx]),
            'stderr_amplitude': np.sqrt(np.sum([out.params['%samplitude' % pfx].stderr**2 for pfx in peak_prefx])),
            'stderr_center': np.sqrt(np.sum([out.params['%scenter' % pfx].stderr**2 for pfx in peak_prefx])) * nn,
            'stderr_sigma': np.sqrt(np.sum([out.params['%ssigma' % pfx].stderr**2 for pfx in peak_prefx])) * nn,
            'stderr_height': np.sqrt(np.sum([out.params['%sheight' % pfx].stderr**2 for pfx in peak_prefx])) * nn,
            'stderr_fwhm': np.sqrt(np.sum([out.params['%sfwhm' % pfx].stderr**2 for pfx in peak_prefx])) * nn,
        }
        for total in totals:
            self.scan.add2namespace(total, totals[total])
        # Background
        self.scan.add2namespace('background', np.mean(comps['bkg_']))

        if print_result:
            print(self.scan.title())
            print(out.fit_report())
            print('Totals:')
            print('\n'.join(self.scan.string(['amplitude', 'center', 'height', 'sigma', 'fwhm', 'background'])))
        if plot_result:
            fig, grid = out.plot()
            fig.suptitle(self.scan.title(), fontsize=12)
            # plt.subplots_adjust(top=0.85, left=0.15)
            ax1, ax2 = fig.axes
            ax1.set_title('')
            ax2.set_xlabel(xname)
            ax2.set_ylabel(yname)
            for component in comps.keys():
                ax2.plot(xdata, comps[component], label=component)
                ax2.legend()
        return out

    def modelfit(self, xaxis='axis', yaxis='signal', model=None, pars=None, method='leastsq',
                 print_result=False, plot_result=False):
        """
        Fit data from scan against lmfit model
        :param xaxis: str name or address of array to plot on x axis
        :param yaxis: str name or address of array to plot on y axis
        :param model: lmfit.Model - object defining combination of models
        :param pars: lmfit.Parameters - object defining model parameters
        :param method: str name of fitting method to use
        :param print_result: bool, if True, print results.fit_report()
        :param plot_result: bool, if True, generate results.plot()
        :return: lmfit fit results

        Example:
            from lmfit.models import GaussianModel, LinearModel
            mod = GaussainModel(prefix='p1_') + LinearModel(prefix='bkg_')
            pars = mod.make_params()
            pars['p1_center'].set(value=np.mean(x), min=x.min(), max=x.max())
            res = scan.fit.modelfit('axis', 'signal', mod, pars)
            print(res.fit_report())
            res.plot()
            area = res.params['p1_amplitude'].value
            err = res.params['p1_amplitude'].stderr
        """
        xdata, ydata, yerror, xname, yname = self.scan.get_plot_data(xaxis, yaxis, None, None)

        # weights
        if yerror is None or np.all(np.abs(yerror) < 0.001):
            weights = None
        else:
            weights = 1 / np.square(yerror, dtype=float)
            weights = np.nan_to_num(weights)

        # Default model, pars
        if model is None:
            model = LinearModel()
        if pars is None:
            pars = model.guess(ydata, x=xdata)

        # lmfit
        res = model.fit(ydata, pars, x=xdata, weights=weights, method=method)

        self.scan.add2namespace('lmfit', res, 'fit_result')
        fit_dict = {}
        for pname, param in res.params.items():
            ename = 'stderr_' + pname
            fit_dict[pname] = param.value
            fit_dict[ename] = param.stderr
        for name, value in fit_dict.items():
            self.scan.add2namespace(name, value)
        self.scan.add2namespace('fit', res.best_fit, other_names=['fit_%s' % yname])
        # Add peak components
        comps = res.eval_components(x=xdata)
        for component in comps.keys():
            self.scan.add2namespace('%sfit' % component, comps[component])

        if print_result:
            print(self.scan.title())
            print(res.fit_report())
        if plot_result:
            fig, grid = res.plot()
            # plt.suptitle(self.title(), fontsize=12)
            # plt.subplots_adjust(top=0.85, left=0.15)
            ax1, ax2 = fig.axes
            ax2.set_xlabel(xname)
            ax2.set_ylabel(yname)
        return res

    def fit_result(self, parameter_name=None):
        """
        Returns parameter, error from the last run fit
        :param parameter_name: str, name from last fit e.g. 'amplitude', or None to return lmfit object
        :param
        :return:
        """
        if 'lmfit' not in self.scan._namespace:
            self.fit()
        lmfit = self.scan('lmfit')
        if parameter_name is None:
            return lmfit
        param = lmfit.params[parameter_name]
        return param.value, param.stderr

    def fit_report(self):
        """Return lmfit.ModelResult.fit_report()"""
        if 'lmfit' not in self.scan._namespace:
            self.fit()
        lmfit = self.scan('lmfit')
        return lmfit.fit_report()

    def plot(self):
        """Plot fit results"""
        if 'lmfit' not in self.scan._namespace:
            self.fit()
        lmfit = self.scan('lmfit')

        fig, grid = lmfit.plot()
        fig.suptitle(self.scan.title(), fontsize=12)
        # plt.subplots_adjust(top=0.85, left=0.15)
        ax1, ax2 = fig.axes
        ax1.set_title('')
        return fig


"----------------------------------------------------------------------------------------------------------------------"
"---------------------------------------------- MultiScanFitManager ---------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"


class MultiScanFitManager:
    """
    MultiScanFitManager
    :param scan: babelscan.MultiScan
    """

    def __init__(self, multiscan):
        self.multiscan = multiscan

    def __call__(self, *args, **kwargs):
        """Calls ScanFitManager.fit(...)"""
        return self.fit(*args, **kwargs)

    def fit(self, xaxis='axes', yaxis='signal', model='Gaussian', background='slope',
            initial_parameters=None, fix_parameters=None, method='leastsq', print_result=False, plot_result=False):
        """
        Automatic fitting of scan

        Use LMFit
        Pass fit_type = LMFit model
        return LMFit output
        """
        out = [
            scan.fit(xaxis, yaxis, model, background, initial_parameters, fix_parameters, method,
                     print_result, plot_result)
            for scan in self.multiscan._scan_list
        ]
        return out

    def multi_peak_fit(self, xaxis='axes', yaxis='signal',
                       npeaks=None, min_peak_power=None, peak_distance_idx=10,
                       model='Gaussian', background='slope',
                       initial_parameters=None, fix_parameters=None, method='leastsq',
                       print_result=False, plot_result=False):
        """
        Automatic fitting of scan

        Use LMFit
        Pass fit_type = LMFit model
        return LMFit output
        """
        out = [
            scan.fit.multi_peak_fit(xaxis, yaxis,
                                    npeaks=npeaks, min_peak_power=min_peak_power, peak_distance_idx=peak_distance_idx,
                                    model=model, background=background, initial_parameters=initial_parameters,
                                    fix_parameters=fix_parameters, method=method,
                                    print_result=print_result, plot_result=plot_result)
            for scan in self.multiscan._scan_list
        ]
        return out

