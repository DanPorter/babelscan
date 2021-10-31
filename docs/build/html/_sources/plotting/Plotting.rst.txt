********
Plotting
********

If you have matplotlib installed, the scan.plot function will automatically be available,
subfunctions of scan.plot include the ability to automatically generate various plots.

Even without matplotlib, it is still possible to generate data for plotting easily using
scan.get_plot_data:

.. code-block:: python

    import babelscan
    scan = babelscan.file_loader('12345.nxs')
    x, y, dy, xlab, ylab = scan.get_plot_data('axes', 'nroi_peak[31,31]', signal_op='/count_time/Transmission', error_op='np.sqrt(x+1)')

    # equivalent to scan.plot()
    plt.figure()
    plt.errorbar(x, y, dy, fmt='-o')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(scan.title())

.. image:: ../images/example_scan_plot.png
    :width: 400
    :alt: example scan plot

You can also easily plot detector images from scan objects:

.. code-block:: python

    import babelscan
    scan = babelscan.file_loader('12345.nxs')

    scan.plot.plot_image(index='sum')

.. image:: ../images/example_scan_image.png
    :width: 400
    :alt: example scan detector image

Plus, if you create a multiscan object, you can make use of automated plotting arguments here too:

.. code-block:: python

    import babelscan
    exp = babelscan.FolderMonitor("/some/folder)
    scans = exp.scans(range(794932, 794947, 1), ['sperp', 'spara'])

    scans.fit.multi_peak_fit(peak_distance_idx=5, print_result=True, plot_result=True)
    scans.plot.plot_simple('sperp', 'amplitude')
    scans.plot.multiplot(yaxis=['signal', 'yfit'])

.. image:: ../images/example_multiscan_multiplot.png
    :width: 400
    :alt: example multiscan multiplot


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   Code
