"""
Matplotlib plotting functions

e.g.
>> set_plot_defaults()
>> axs = create_multiplot(2, 2, title='New figure')
>> plot_line(axs[0], x, y)
>> plot_line(axs[1], x2, y2, label='data 2')
>> plot_lines(ax[2], x, [y, y2])
>> plot_detector_image(ax[3], image)
>> labels('title', 'x', 'y', legend=True, axes=axs[0])
"""

import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # 3D plotting

from . import functions as fn


DEFAULT_FONT = 'Times New Roman'
DEFAULT_FONTSIZE = 14
FIG_SIZE = [12, 8]
FIG_DPI = 80


'----------------------------Plot manipulation--------------------------'


def set_plot_defaults(rcdefaults=False):
    """
    Set custom matplotlib rcparams, or revert to matplotlib defaults
    These handle the default look of matplotlib plots
    See: https://matplotlib.org/stable/tutorials/introductory/customizing.html#the-default-matplotlibrc-file
    :param rcdefaults: False*/ True, if True, revert to matplotlib defaults
    :return: None
    """
    if rcdefaults:
        print('Return matplotlib rcparams to default settings.')
        plt.rcdefaults()
        return

    plt.rc('figure', figsize=FIG_SIZE, dpi=FIG_DPI, autolayout=False)
    plt.rc('lines', marker='o', color='r', linewidth=2, markersize=6)
    plt.rc('errorbar', capsize=2)
    plt.rc('legend', loc='best', frameon=False, fontsize=DEFAULT_FONTSIZE)
    plt.rc('axes', linewidth=2, titleweight='bold', labelsize='large')
    plt.rc('xtick', labelsize='large')
    plt.rc('ytick', labelsize='large')
    plt.rc('axes.formatter', limits=(-3, 3), offset_threshold=6)
    plt.rc('image', cmap='viridis')  # default colourmap, see https://matplotlib.org/stable/gallery/color/colormap_reference.html
    # Note font values appear to only be set when plt.show is called
    plt.rc(
        'font',
        family='serif',
        style='normal',
        weight='bold',
        size=DEFAULT_FONTSIZE,
        serif=['Times New Roman', 'Times', 'DejaVu Serif']
    )
    # plt.rcParams["savefig.directory"] = os.path.dirname(__file__) # Default save directory for figures


def labels(ttl=None, xvar=None, yvar=None, zvar=None, legend=False,
           colorbar=False, colorbar_label=None,
           axes=None, size=None):
    """
    Add formatted labels to current plot, also increases the tick size
    :param ttl: title
    :param xvar: x label
    :param yvar: y label
    :param zvar: z label (3D plots only)
    :param legend: False/ True, adds default legend to plot
    :param colorbar: False/ True, adds default colorbar to plot
    :param colorbar_label: adds label to colorbar
    :param axes: matplotlib axes to use, None for plt.gca()
    :param size: None, 'Normal' or 'Big'
    :param font: str font name, 'Times New Roman'
    :return: None
    """
    if axes is None:
        axes = plt.gca()

    if size is not None:
        if size.lower() in ['big', 'large', 'xxl', 'xl']:
            tik = 30
            tit = 32
            lab = 35
        elif size.lower() in ['small', 'tiny']:
            tik = 8
            tit = 12
            lab = 8
    else:
        # Normal
        tik = None
        tit = None
        lab = None

    axes.tick_params(axis="x", labelsize=tik)
    axes.tick_params(axis="y", labelsize=tik)

    if ttl is not None:
        axes.set_title(ttl, fontsize=tit, fontweight='bold')

    if xvar is not None:
        axes.set_xlabel(xvar, fontsize=lab)

    if yvar is not None:
        axes.set_ylabel(yvar, fontsize=lab)

    if zvar is not None:
        # Don't think this works, use ax.set_zaxis
        axes.set_zlabel(zvar, fontsize=lab)

    if legend:
        axes.legend(fontsize=lab)

    if colorbar:
        mappables = axes.images + axes.collections
        cb = plt.colorbar(mappables[0], ax=axes)
        if colorbar_label:
            cb.set_ylabel(colorbar_label, fontsize=lab)


def colormap(clim=None, cmap=None, axes=None):
    """
    Set colour limits and colormap on axes
    :param clim: [min, max] color cut-offs
    :param cmap: str name of colormap. e.g. 'viridis' or 'spectral'
    :param axes: matplotlib axes or None for current axes
    :return: None
    """
    if axes is None:
        axes = plt.gca()

    # Get axes images
    mappables = axes.images + axes.collections
    for image in mappables:
        if cmap:
            image.set_cmap(plt.get_cmap(cmap))
        if clim:
            image.set_clim(clim)


def saveplot(name, dpi=None, figure_number=None):
    """
    Saves current figure as a png in the home directory
    :param name: filename, including or expluding directory and or extension
    :param dpi: image resolution, higher means larger image size, default=matplotlib default
    :param figure_number: figure number, default = plt.gcf()
    :return: None

    E.G.
    ---select figure to save by clicking on it---
    saveplot('test')
    E.G.
    saveplot('c:\somedir\apicture.jpg', dpi=600, figure=3)
    """
    if figure_number is None:
        gcf = plt.gcf()
    else:
        gcf = plt.figure(figure_number)

    filedir = os.path.dirname(name)
    file, ext = os.path.splitext(name)

    if filedir is None:
        filedir = os.path.expanduser('~')

    if len(ext) == 0:
        ext = '.png'

    savefile = os.path.join(filedir, file + ext)
    gcf.savefig(savefile, dpi=dpi)
    print('Saved Figure {} as {}'.format(gcf.number, savefile))


def create_axes(fig=None, subplot=111, *args, **kwargs):
    """
    Create new plot axis
    ax = create_axes(subplot=111)

    for 3D plot, use: create_axes(projection='3d')

    :param fig: matplotlib figure object, or None to create Figure
    :param subplot: subplot input
    :param *args, **kwargs: pass additional argments to fig.add_subplot
    :return: axes object
    """
    set_plot_defaults()
    if fig is None:
        fig = plt.figure()
    ax = fig.add_subplot(subplot, *args, **kwargs)
    return ax


def create_multiplot(nrows=5, ncols=5, title=None, **kwargs):
    """
    Create large figure with multiple plots
    :param nrows: int number of figure rows
    :param ncols: int number of figure columns
    :param title: str suptitle
    :param kwargs: arguments to pass to plt.subplots
    :return: list of axes, len nrows*ncols
    """
    set_plot_defaults()
    if 'figsize' not in kwargs:
        kwargs['figsize'] = [FIG_SIZE[0] * 2, FIG_SIZE[1] * 2]
    if 'dpi' not in kwargs:
        kwargs['dpi'] = FIG_DPI
    fig, ax = plt.subplots(nrows, ncols, **kwargs)
    fig.subplots_adjust(hspace=0.35, wspace=0.32, left=0.07, right=0.97)
    if title is not None:
        plt.suptitle(title, fontsize=22)
    return np.reshape(ax, -1)


def plot_line(axes, xdata, ydata, yerrors=None, line_spec='-o', *args, **kwargs):
    """
    Plot line on given matplotlib axes subplot
    Uses matplotlib.plot or matplotlib.errorbar if yerrors is not None
    :param axes: matplotlib figure or subplot axes, None uses current axes
    :param xdata: array data on x axis
    :param ydata: array data on y axis
    :param yerrors: array errors on y axis (or None)
    :param line_spec: str matplotlib.plot line_spec
    :param args: additional arguments
    :param kwargs: additional arguments
    :return: output of plt.plot [line], or plt.errorbar [line, xerrors, yerrors]
    """
    if axes is None:
        axes = plt.gca()

    if yerrors is None:
        lines = axes.plot(xdata, ydata, line_spec, *args, **kwargs)
    else:
        lines = axes.errorbar(xdata, ydata, yerrors, *args, fmt=line_spec, **kwargs)
    return lines


def plot_lines(axes, xdata, ydata, yerrors=None, cdata=None, cmap=None, line_spec='-o', *args, **kwargs):
    """
    Plot lines on given matplotlib axes subplot
    Uses matplotlib.plot or matplotlib.errorbar if yerrors is not None
    :param axes: matplotlib figure or subplot axes, None uses current axes
    :param xdata: array[n] data on x axis
    :param ydata: list[n] of array[n] data on y axis
    :param yerrors: list[m] of array[n] errors on y axis (or None)
    :param cdata: list[n] of values to define line colour
    :param cmap: name of colormap to generate colour variation in lines
    :param line_spec: str or list[m] of str matplotlib.plot line_spec
    :param args: additional arguments
    :param kwargs: additional arguments
    :return: output of plt.plot [line], or plt.errorbar [line, xerrors, yerrors]
    """
    if axes is None:
        axes = plt.gca()

    nplots = len(ydata)
    if xdata is None:
        xdata = [range(len(y)) for y in ydata]
    elif len(xdata) != nplots:
        xdata = [xdata] * nplots

    if yerrors is None:
        yerrors = [None] * nplots
    elif len(yerrors) != nplots:
        yerrors = [yerrors] * nplots

    if cmap is None:
        cmap = 'viridis'
    if cdata is None:
        cdata = np.arange(nplots)
    else:
        cdata = np.asarray(cdata)
    cnorm = cdata - cdata.min()
    cnorm = cnorm / cnorm.max()
    cols = plt.get_cmap(cmap)(cnorm)

    line_spec = fn.liststr(line_spec)
    if len(line_spec) != nplots:
        line_spec = line_spec * nplots

    print(axes)
    print(len(xdata), xdata)
    print(len(ydata), ydata)
    print(len(yerrors), yerrors)
    print(len(line_spec), line_spec)
    print(len(cols), cols)

    lines = []
    for n in range(nplots):
        lines += plot_line(axes, xdata[n], ydata[n], yerrors[n], line_spec[n], c=cols[n], *args, **kwargs)
    return lines


def plot_detector_image(axes, image, clim=None, *args, **kwargs):
    """
    Plot detector image
    :param axes: matplotlib figure or subplot axes, None uses current axe
    :param image: 2d array image data
    :param clim: None or [min, max] values for color cutoff
    :param args: additional arguments for plt.pcolormesh
    :param kwargs: additional arguments for plt.pcolormesh
    :return: axes object
    """
    if axes is None:
        axes = plt.gca()

    if 'shading' not in kwargs.keys():
        kwargs['shading'] = 'gouraud'
    if clim:
        kwargs['vmin'] = clim[0]
        kwargs['vmax'] = clim[1]

    axes.pcolormesh(image, *args, **kwargs)
    axes.invert_yaxis()
    axes.axis('image')
    return axes


def plot_2d_surface(axes, image, xdata=None, ydata=None, clim=None, axlim='image', **kwargs):
    """
    Plot 2D data as colourmap surface
    :param axes: matplotlib figure or subplot axes, None uses current axe
    :param image: 2d array image data
    :param xdata: array data, 2d or 1d
    :param ydata: array data 2d or 1d
    :param clim: None or [min, max] values for color cutoff from plt.clim
    :param axlim: axis limits from plt.axis
    :param kwargs: additional arguments for plt.pcolormesh
    :return: output of plt.pcolormesh
    """
    if axes is None:
        axes = plt.gca()

    if 'shading' not in kwargs.keys():
        kwargs['shading'] = 'gouraud'
    if clim:
        kwargs['vmin'] = clim[0]
        kwargs['vmax'] = clim[1]
    if np.ndim(xdata) == 1 and np.ndim(ydata) == 1:
        ydata, xdata = np.meshgrid(ydata, xdata)

    if xdata:
        surface = axes.pcolormesh(xdata, ydata, image, **kwargs)
    else:
        surface = axes.pcolormesh(image, **kwargs)
    axes.axis(axlim)
    return surface


def plot_3d_surface(axes, image, xdata=None, ydata=None, samples=None, clim=None, axlim='auto', **kwargs):
    """
    Plot 2D image data as 3d surface
    :param axes: matplotlib figure or subplot axes, None uses current axe
    :param image: 2d array image data
    :param xdata: array data, 2d or 1d
    :param ydata: array data 2d or 1d
    :param samples: max number of points to take in each direction, by default does not downsample
    :param clim: None or [min, max] values for color cutoff from plt.clim
    :param axlim: axis limits from plt.axis
    :param kwargs: additional arguments for plt.plot_surface
    :return: output of plt.plot_surface
    """
    if axes is None:
        axes = plt.gca()

    if samples:
        kwargs['rcount'] = samples
        kwargs['ccount'] = samples
    else:
        # default in plot_surface is 50
        kwargs['rcount'],  kwargs['ccount'] = np.shape(image)
    if clim:
        kwargs['vmin'] = clim[0]
        kwargs['vmax'] = clim[1]

    if np.ndim(xdata) == 1 and np.ndim(ydata) == 1:
        ydata, xdata = np.meshgrid(ydata, xdata)

    if xdata:
        surface = axes.plot_surface(xdata, ydata, image, **kwargs)
    else:
        surface = axes.plot_surface(image, **kwargs)
    axes.axis(axlim)
    return surface


def plot_image_slider(image=None, yerrors=None, clim=None):
    """
    Create matplotlib figure with x,y plot on left and detector image on write
    :param image:
    :param yerrors:
    :param clim:
    :return:
    """
    pass


def create_html_page(body_lines=(), header_lines=()):
    """Create html page"""
    css = """<style>
    .ScanBox {
      border: 1px solid black;
      padding-top: 10px;
      padding-bottom: 10px;
      padding-left: 10px;
      width: 90pc;
      height: 400px;
      resize: vertical;
    }
    .ScanDetails {
      float: left;
      width: 40pc;
    }
    .ScanImage{
      float: left:
      padding-left:2px;
      width: 22pc;
    }
  </style>"""
    header_lines = css.splitlines() + list(header_lines)

    html = "<!doctype html>\n<html lang=\"en\">\n<html>\n\n"
    html += "<head>\n  %s\n</head>\n\n" % '\n  '.join(header_lines)
    html += "<body>\n  %s\n</body>" % '\n  '.join(body_lines)
    html += "\n\n</html>\n"
    return html


def create_figure_div(title, details, fig1_file, fig2_file=None):
    """
    Create html code to generate scan details div
    :param title: str title (single line)
    :param details: str details of scan (multi-line)
    :param fig1_file: str
    :param fig2_file: str or None
    :param class_name: str
    :return:
    """

    detail_div = ["  <div class=\"ScanDetails\">"]
    # detail_div += ['    <p>%s</p>' % det for det in details.splitlines()]
    detail_div += ['    %s' % details.replace('\n', '<br>')]
    detail_div += ["  </div>"]

    image1 = "  <img src=\"%s\" alt=\"%s\" class=\"ScanImage\">" % (
            fig1_file, title)
    if fig2_file is None:
        image2 = ""
    else:
        image2 = "  <img src=\"%s\" alt=\"%s\" class=\"ScanImage\">" % (
            fig2_file, title)

    html = [
        "<div class=\"ScanBox\">",
        "  <h3>%s</h3>" % title.replace('\n', '<br>'),
    ]
    html = html + detail_div
    html += [
        image1,
        image2,
        "</div>"
        ""
    ]
    return html


def create_plotly_blob(data_list, xlabel, ylabel, title):
    """
    Create plotly line plot object, useful for jupyter plots or generation of interactive html plots
    E.G.
      import plotly.graph_objects as go
      blob = create_plotly_blob([(xdata1, ydata1, label1, True), (xdata2, ydata2, label2, False)], 'x', 'y', 'title')
      fig = go.Figure(blob)
      fig.show()

    4 element tuples in data_list must follow (xdata, ydata, label, visible):
       xdata: 1d array of x-axis data
       ydata: 1d array of y-axis data
       label: str label description
       visible: bool, if False plot is only given in legend but not turned on
    :param data_list: list of 4 element tuples (xdata, ydata, label, visible)
    :param xlabel: str x-axis label
    :param ylabel: str y-axis label
    :param title: str plot title
    :return: dict
    """
    # Convert title
    title = title.replace('\n', '<br>')

    # Create json blob
    auto_blob = {
        'data': [],
        'layout': {'font': {'family': 'Courier New, monospace', 'size': 18},
                   'legend': {'title': {'text': 'Scannables'}},
                   'title': {'text': title},
                   'xaxis': {'title': {'text': xlabel}},
                   'yaxis': {'title': {'text': ylabel}}}
    }

    for item in data_list:
        if not item[3]:
            vis = 'legendonly'
        else:
            vis = True
        trace = {
            'mode': 'markers+lines',
            'name': item[2],
            'type': 'scatter',
            'visible': vis,
            'x': list(item[0]),
            'y': list(item[1]),

        }
        auto_blob['data'] += [trace]
    return auto_blob


"----------------------------------------------------------------------------------------------------------------------"
"----------------------------------------------- ScanPlotManager ------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"


class ScanPlotManager:
    """
    ScanPlotManager
        scan.plot = ScanPlotManager(scan)
        scan.plot() # plot default axes
        scan.plot.plot(xaxis, yaxis)  # creates figure
        scan.plot.plotline(xaxis, yaxis)  # plots line on current figure
        scan.plot.image()  # create figure and display detector image

    Options called from babelscan.Scan:
      'plot_show': True >> automatically call "plt.show" after plot command

    :param scan: babelscan.Scan
    """
    def __init__(self, scan):
        self.scan = scan

    def __call__(self, *args, **kwargs):
        """Calls ScanPlotManager.plot(...)"""
        return self.plot(*args, **kwargs)

    show = plt.show

    def plotline(self, xaxis='axes', yaxis='signal', *args, **kwargs):
        """
        Plot scanned datasets on matplotlib axes subplot
        :param xaxis: str name or address of array to plot on x axis
        :param yaxis: str name or address of array to plot on y axis
        :param args: given directly to plt.plot(..., *args, **kwars)
        :param axes: matplotlib.axes subplot, or None to use plt.gca()
        :param kwargs: given directly to plt.plot(..., *args, **kwars)
        :return: list lines object, output of plot
        """
        xdata, ydata, yerror, xname, yname = self.scan.get_plot_data(xaxis, yaxis, None, None)

        if 'label' not in kwargs:
            kwargs['label'] = self.scan.label()
        axes = kwargs.pop('axes') if 'axes' in kwargs else None
        lines = plot_line(axes, xdata, ydata, None, *args, **kwargs)
        return lines

    def plot(self, xaxis='axes', yaxis='signal', *args, **kwargs):
        """
        Create matplotlib figure with plot of the scan
        :param axes: matplotlib.axes subplot
        :param xaxis: str name or address of array to plot on x axis
        :param yaxis: str name or address of array to plot on y axis, also accepts list of names for multiplt plots
        :param args: given directly to plt.plot(..., *args, **kwars)
        :param axes: matplotlib.axes subplot, or None to create a figure
        :param kwargs: given directly to plt.plot(..., *args, **kwars)
        :return: axes object
        """
        # Check for multiple inputs on yaxis
        ylist = fn.liststr(yaxis)

        # Create figure
        if 'axes' in kwargs:
            axes = kwargs.pop('axes')
        else:
            axes = create_axes(subplot=111)

        xname, yname = xaxis, yaxis
        for yaxis in ylist:
            xdata, ydata, yerror, xname, yname = self.scan.get_plot_data(xaxis, yaxis, None, None)
            plot_line(axes, xdata, ydata, None, *args, label=yname, **kwargs)

        # Add labels
        ttl = self.scan.title()
        labels(ttl, xname, yname, legend=True, axes=axes)
        if self.scan.options('plot_show'):
            plt.show()
        return axes

    def image(self, index=None, xaxis='axes', axes=None, clim=None, cmap=None, colorbar=False, **kwargs):
        """
        Plot image in matplotlib figure (if available)
        :param index: int, detector image index, 0-length of scan, if None, use centre index
        :param xaxis: name or address of xaxis dataset
        :param axes: matplotlib axes to plot on (None to create figure)
        :param clim: [min, max] colormap cut-offs (None for auto)
        :param cmap: str colormap name (None for auto)
        :param colorbar: False/ True add colorbar to plot
        :param kwargs: additinoal arguments for plot_detector_image
        :return: axes object
        """
        # x axis data
        xname, xdata = self.scan._name_eval(xaxis)

        # image data
        im = self.scan.image(index)
        if index is None or index == 'sum':
            xvalue = xdata[np.size(xdata) // 2]
        else:
            xvalue = xdata[index]

        # Create figure
        if axes is None:
            axes = create_axes(subplot=111)
        plot_detector_image(axes, im, **kwargs)

        # labels
        ttl = '%s\n%s [%s] = %s' % (self.scan.title(), xname, index, xvalue)
        labels(ttl, colorbar=colorbar, colorbar_label='Detector', axes=axes)
        colormap(clim, cmap, axes)
        if self.scan.options('plot_show'):
            plt.show()
        return axes

    def image_slider(self, index=None, xaxis='axes', axes=None, clim=None, cmap=None, colorbar=False, **kwargs):
        """
        Plot image in matplotlib figure (if available)
        :param index: int, detector image index, 0-length of scan, if None, use centre index
        :param xaxis: name or address of xaxis dataset
        :param axes: matplotlib axes to plot on (None to create figure)
        :param clim: [min, max] colormap cut-offs (None for auto)
        :param cmap: str colormap name (None for auto)
        :param colorbar: False/ True add colorbar to plot
        :param kwargs: additinoal arguments for plot_detector_image
        :return: axes object
        """
        # x axis data
        xname, xdata = self.scan._name_eval(xaxis)

        # image data
        vol = self.scan.volume()

        # Create figure
        if axes is None:
            axes = create_axes(subplot=111)

        vol.plot.image_slider(index=index, axes=axes, clim=clim, cmap=cmap, colorbar=colorbar, **kwargs)

        # labels
        ttl = '%s\n%s' % (self.scan.title(), xname)
        labels(ttl, colorbar=colorbar, colorbar_label='Detector', axes=axes)
        colormap(clim, cmap, axes)
        if self.scan.options('plot_show'):
            plt.show()
        return axes

    def detail(self, xaxis='axes', yaxis='signal', index=None, clim=None, cmap=None, **kwargs):
        """
        Create matplotlib figure with plot of the scan and detector image
        :param axes: matplotlib.axes subplot
        :param xaxis: str name or address of array to plot on x axis
        :param yaxis: str name or address of array to plot on y axis, also accepts list of names for multiplt plots
        :param index: int, detector image index, 0-length of scan, if None, use centre index
        :param xaxis: name or address of xaxis dataset
        :param clim: [min, max] colormap cut-offs (None for auto)
        :param cmap: str colormap name (None for auto)
        :param kwargs: given directly to plt.plot(..., *args, **kwars)
        :return: axes object
        """

        # Create figure
        fig, ((lt, rt), (lb, rb)) = plt.subplots(2, 2, figsize=[FIG_SIZE[0] * 1.2, FIG_SIZE[1] * 1.2], dpi=FIG_DPI)
        fig.subplots_adjust(hspace=0.35, left=0.1, right=0.95)

        # Top left - line plot
        self.plot(xaxis, yaxis, axes=lt, **kwargs)

        # Top right - image plot
        try:
            self.image(index, xaxis, cmap=cmap, clim=clim, axes=rt)
        except (FileNotFoundError, KeyError, TypeError):
            rt.text(0.5, 0.5, 'No Image')
            rt.set_axis_off()

        # Bottom-Left - details
        details = str(self.scan)
        lb.text(-0.1, 0.8, details, multialignment="left", fontsize=12, wrap=True)
        lb.set_axis_off()

        rb.set_axis_off()

        if self.scan.options('plot_show'):
            plt.show()
        return fig

    def plotly_blob(self, xaxis='axes', yaxis='signal'):
        """
        Create plotly line plot object, useful for jupyter plots or generation of interactive html plots
        E.G.
          import plotly.graph_objects as go
          blob = scan.plot.plotly_blob('axes', ['signal', 'signal/2'])
          fig = go.Figure(blob)
          fig.show()
        :param xaxis: str name or address of array to plot on x axis
        :param yaxis: str name or address of array to plot on y axis, also accepts list of names for multiplt plots
        :return: dict
        """
        # Check for multiple inputs on yaxis
        ylist = fn.liststr(yaxis)

        xname, yname = xaxis, yaxis
        data_list = []
        for yaxis in ylist:
            xdata, ydata, yerror, xname, yname = self.scan.get_plot_data(xaxis, yaxis, None, None)
            data_list += [(xdata, ydata, yname, True)]
        ttl = self.scan.title()
        return create_plotly_blob(data_list, xname, yname, ttl)


"----------------------------------------------------------------------------------------------------------------------"
"-------------------------------------------- MultiScanPlotManager ----------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"


class MultiScanPlotManager:
    """
    ScanPlotManager
    :param scan: babelscan.Scan
    """
    def __init__(self, multiscan):
        self.multiscan = multiscan

    def __call__(self, *args, **kwargs):
        return self.plot(*args, **kwargs)

    show = plt.show

    def plot_simple(self, xname, yname, *args, **kwargs):
        """
        Simple plot method, retrieves x,y data and plots using plt.plot
        :param xname:
        :param yname:
        :param args, kwargs: same as plt.plot(x,y, ...)
        :return: axis
        """

        # Get data
        xdata, ydata, xlabel, ylabel = self.multiscan.get_plot_data(xname, yname)

        # Create figure
        if 'axes' in kwargs:
            axes = kwargs.pop('axes')
        else:
            axes = create_axes(subplot=111)

        axes.plot(xdata, ydata, *args, **kwargs)
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)
        axes.set_title(self.multiscan.title())

        # Add legend if multiple arrays added
        if np.ndim(xdata[0]) > 0:
            # xdata is a list of arrays
            scan_labels = self.multiscan.labels()
            axes.legend(scan_labels)
        return axes

    def plot(self, xaxis='axes', yaxis='signal', *args, **kwargs):
        """
        Create matplotlib figure with plot of the scan
        :param axes: matplotlib.axes subplot
        :param xaxis: str name or address of array to plot on x axis
        :param yaxis: str name or address of array to plot on y axis
        :param args: given directly to plt.plot(..., *args, **kwars)
        :param axes: matplotlib.axes subplot, or None to create a figure
        :param kwargs: given directly to plt.plot(..., *args, **kwars)
        :return: axes object
        """

        # Create figure
        if 'axes' in kwargs:
            axes = kwargs.pop('axes')
        else:
            axes = create_axes(subplot=111)

        xname, yname = xaxis, yaxis
        scan_labels = self.multiscan.labels()
        for n, scan in enumerate(self.multiscan):
            xdata, ydata, yerror, xname, yname = scan.get_plot_data(xaxis, yaxis, None, None)
            plot_line(axes, xdata, ydata, None, *args, label=scan_labels[n], **kwargs)

        # Add labels
        ttl = self.multiscan.title()
        labels(ttl, xname, yname, legend=True)
        if self.multiscan[0].options('plot_show'):
            plt.show()
        return axes

    def multiplot(self, xaxis='axes', yaxis='signal', size=(4, 4), *args, **kwargs):
        """
        Create matplotlib figure with plot of the scan
        :param axes: matplotlib.axes subplot
        :param xaxis: str name or address of array to plot on x axis
        :param yaxis: str name or address of array to plot on y axis
        :param size: (ncol, nrow) int values number of axes
        :param kwargs: given directly to plt.plot(..., *args, **kwars)
        :return: axes object
        """

        n_axes = size[0] * size[1]
        n_figs = int(np.ceil(len(self.multiscan)/float(n_axes)))
        scan_labels = self.multiscan.labels()
        yaxis_list = fn.liststr(yaxis)
        for figno in range(n_figs):
            scans = self.multiscan[figno * n_axes: (figno + 1) * n_axes]
            scan_labels = scan_labels[figno * n_axes: (figno + 1) * n_axes]

            # Create figure
            axes = create_multiplot(size[0], size[1])
            for n, scan in enumerate(scans):
                for yaxis in yaxis_list:
                    xdata, ydata, yerror, xname, yname = scan.get_plot_data(xaxis, yaxis, None, None)
                    plot_line(axes[n], xdata, ydata, None, *args, label=yname, **kwargs)
                    labels(scan_labels[n], xaxis, axes=axes[n], size='tiny', legend=True)

    def plot_details_to_pdf(self, filename):
        """
        Create pdf file with scans
        :param filename: str pdf filename
        :return: None
        """

        filename, ext = os.path.splitext(filename)
        filename = filename + '.pdf'

        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(filename) as pdf:
            for scan, label in zip(self.multiscan, self.multiscan.labels()):
                fig = scan.plot.detail()
                fig.suptitle(label)
                pdf.savefig(fig)
                plt.close(fig)
            # PDF metadata
            d = pdf.infodict()
            d['Title'] = self.multiscan.title()
            d['Author'] = 'BabelScan'
            d['Subject'] = 'Created in BabelScan.MultiScan.MultiScanPlotManager.plot_details_to_pdf'
            d['Keywords'] = 'BabelScan'
            d['CreationDate'] = datetime.datetime.now()
            d['ModDate'] = datetime.datetime.now()

    def plot_details_to_html(self, folder_name):
        """
        Create html file with scans
        :param filename: str name of folder to create
        :return: None
        """
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)
            print('Created Folder: %s' % folder_name)

        html = []
        for scan, label in zip(self.multiscan, self.multiscan.labels()):
            scan.plot.plot()
            fname1 = folder_name + '/%s.svg' %  scan.scan_number
            plt.savefig(fname1)
            plt.close()
            fname1 = '%s.svg' % scan.scan_number
            try:
                scan.plot.image()
                fname2 = folder_name + '/%s.png' % scan.scan_number
                plt.savefig(fname2)
                plt.close()
                fname2 = '%s.png' % scan.scan_number
            except (FileNotFoundError, KeyError, TypeError):
                fname2 = None

            title = scan.title()
            details = str(scan)
            html += create_figure_div(title, details, fname1, fname2)

        html_page = create_html_page(html)
        with open(folder_name + '/index.html', 'wt') as f:
            f.write(html_page)
        print('Scans written to %s' % (folder_name + '/index.html'))

    def plotly_blob(self, xaxis='axes', yaxis='signal'):
        """
        Create plotly line plot object, useful for jupyter plots or generation of interactive html plots
        E.G.
          import plotly.graph_objects as go
          blob = scans.plot.plotly_blob('axes', 'signal')
          fig = go.Figure(blob)
          fig.show()
        :param xaxis: str name or address of array to plot on x axis
        :param yaxis: str name or address of array to plot on y axis
        :return: dict
        """
        xname, yname = xaxis, yaxis
        data_list = []
        scan_labels = self.multiscan.labels()
        for n, scan in enumerate(self.multiscan):
            xdata, ydata, yerror, xname, yname = scan.get_plot_data(xaxis, yaxis, None, None)
            data_list += [(xdata, ydata, scan_labels[n], True)]
        ttl = self.multiscan.title()
        return create_plotly_blob(data_list, xname, yname, ttl)


"----------------------------------------------------------------------------------------------------------------------"
"----------------------------------------------- VolumePlotManager ----------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"


class VolumePlotManager:
    """
    VolumePlotManager
        vol.plot = VolumePlotManager(vol)
        vol.plot() #
        vol.plot.image()  # create figure and display detector image

    Options called from babelscan.volume.Volume:
      'plot_show': True >> automatically call "plt.show" after plot command

    :param vol: babelscan.volume.Volume
    """
    def __init__(self, vol):
        self.vol = vol

    def __call__(self, *args, **kwargs):
        """Calls ScanPlotManager.plot(...)"""
        return self.image(*args, **kwargs)

    show = plt.show

    def image(self, index=None, axis=0, clim=None, cmap=None, colorbar=False, **kwargs):
        """
        Plot image in matplotlib figure (if available)
        :param index: int, image index, 0-length of scan, if None, use centre index
        :param axis: int, axis to index (0-2)
        :param clim: [min, max] colormap cut-offs (None for auto)
        :param cmap: str colormap name (None for auto)
        :param colorbar: False/ True add colorbar to plot
        :param kwargs: additinoal arguments for plot_detector_image
        :return: axes object
        """

        if index is None:
            index = self.vol.shape[axis] // 2

        if index == 'sum':
            im = np.sum(self.vol, axis=axis)
        elif axis == 1:
            im = self.vol[:, index, :]
        elif axis == 2:
            im = self.vol[:, :, index]
        else:
            im = self.vol[index]

        # Create figure
        show = False
        if 'axes' not in kwargs:
            show = True
            axes = create_axes(subplot=111)
        else:
            axes = kwargs.pop('axes')
        plot_detector_image(axes, im, **kwargs)

        # labels
        ttl = 'volume[%s]' % index
        xlab = '[%s, 0, :]' % index  # might be wrong way around
        ylab = '[%s, :, 0]' % index  # might be wrong way around
        labels(ttl, xlab, ylab, colorbar=colorbar, colorbar_label='Detector', axes=axes)
        colormap(clim, cmap, axes)
        if show:
            plt.show()
        return axes

    def image_slider(self, index=None, axis=0, clim=None, cmap=None, colorbar=False, **kwargs):
        """
        Plot image in matplotlib figure with a slider (if available)
        :param index: int, image index, 0-length of scan, if None, use centre index
        :param axis: int, axis to index (0-2)
        :param clim: [min, max] colormap cut-offs (None for auto)
        :param cmap: str colormap name (None for auto)
        :param colorbar: False/ True add colorbar to plot
        :param kwargs: additinoal arguments for plot_detector_image
        :return: axes object
        """
        if index is None:
            index = self.vol.shape[axis] // 2

        # Create figure
        show = False
        if 'axes' not in kwargs:
            show = True
            axes = create_axes(subplot=111)
            kwargs['axes'] = axes
        axes = self.image(index, axis, clim, cmap, colorbar, **kwargs)

        # pcolormesh object
        pcolor = axes.collections[0]

        # Move axes for slider
        bbox = axes.get_position()
        left, bottom, width, height = bbox.x0, bbox.y0, bbox.width, bbox.height
        change_in_height = height * 0.1
        new_position = [left, bottom + 2 * change_in_height, width, height - 2 * change_in_height]
        new_axes_position = [left, bottom, width, change_in_height]

        axes.set_position(new_position, 'original')
        new_axes = axes.figure.add_axes(new_axes_position)

        sldr = plt.Slider(new_axes, 'Volume', 0, self.vol.shape[axis], valinit=index, valfmt='%0.0f')

        def update(val):
            """Update function for pilatus image"""
            imgno = int(round(sldr.val))
            if axis == 1:
                im = self.vol[:, imgno, :]
            elif axis == 2:
                im = self.vol[:, :, imgno]
            else:
                im = self.vol[imgno]
            pcolor.set_array(im.flatten())
            plt.draw()
            # fig.canvas.draw()

        sldr.on_changed(update)

        if show:
            plt.show()
        return axes

    def axis_sum(self, sum_axis=0, *args, **kwargs):
        """
        Plot cut along axis, summed in other 2 axes
        :param sum_axis: axis to plot (0-2), summed in other axes
        :param args: given directly to plt.plot(..., *args, **kwars)
        :param axes: matplotlib.axes subplot, or None to create a figure
        :param kwargs: given directly to plt.plot(..., *args, **kwars)
        :return: axes object
        """

        if sum_axis == 1:
            out = np.sum(np.sum(self.vol, axis=0), axis=1)
        elif sum_axis == 2:
            out = np.sum(np.sum(self.vol, axis=0), axis=0)
        else:
            out = np.sum(np.sum(self.vol, axis=1), axis=1)

        if 'label' not in kwargs:
            kwargs['label'] = 'axis %d' % sum_axis
        axes = kwargs.pop('axes') if 'axes' in kwargs else None
        lines = plot_line(axes, range(len(out)), out, None, *args, **kwargs)
        return lines

    def cut(self, index1=(0, 0, 0), index2=(-1, -1, -1), *args, **kwargs):
        """
        Plot arbitary cut through the volume from index1 to index2
        :param index1: (i,j,k) start point in the volume
        :param index2: (i,j,k) end point in the volume
        :param args: given directly to plt.plot(..., *args, **kwars)
        :param axes: matplotlib.axes subplot, or None to create a figure
        :param kwargs: given directly to plt.plot(..., *args, **kwars)
        :return: axes object
        """
        volcut = self.vol.cut(index1, index2)
        if 'label' not in kwargs:
            kwargs['label'] = '%s-%s' % (tuple(index1), tuple(index2))
        axes = kwargs.pop('axes') if 'axes' in kwargs else None
        lines = plot_line(axes, range(len(volcut)), volcut, None, *args, **kwargs)
        return lines

    def array_sum(self, *args, **kwargs):
        """Plots [sum(image) for image in volume]"""

        out = self.vol.array_sum()

        if 'label' not in kwargs:
            kwargs['label'] = 'array sum'
        axes = kwargs.pop('axes') if 'axes' in kwargs else None
        lines = plot_line(axes, range(len(out)), out, None, *args, **kwargs)
        return lines

    def array_max(self, *args, **kwargs):
        """Plots [max(image) for image in volume]"""

        out = self.vol.array_max()

        if 'label' not in kwargs:
            kwargs['label'] = 'array max'
        axes = kwargs.pop('axes') if 'axes' in kwargs else None
        lines = plot_line(axes, range(len(out)), out, None, *args, **kwargs)
        return lines

    def array_min(self, *args, **kwargs):
        """Plots [min(image) for image in volume]"""

        out = self.vol.array_min()

        if 'label' not in kwargs:
            kwargs['label'] = 'array min'
        axes = kwargs.pop('axes') if 'axes' in kwargs else None
        lines = plot_line(axes, range(len(out)), out, None, *args, **kwargs)
        return lines

    def array_mean(self, *args, **kwargs):
        """Plots [mean(image) for image in volume]"""

        out = self.vol.array_mean()

        if 'label' not in kwargs:
            kwargs['label'] = 'array mean'
        axes = kwargs.pop('axes') if 'axes' in kwargs else None
        lines = plot_line(axes, range(len(out)), out, None, *args, **kwargs)
        return lines

