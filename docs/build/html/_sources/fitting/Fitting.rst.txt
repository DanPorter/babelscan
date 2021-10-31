*******
Fitting
*******

Automated peak fitting functions using lmfit. If lmfit is installed, scan objects will
automatically contain the scan.fit class with various automated methods.

See: https://lmfit.github.io/lmfit-py/builtin_models.html

.. code-block:: python

    from babelscan.fitting import multipeakfit
    fit = multipeakfit(xdata, ydata)  # returns lmfit object
    print(fit)
    fit.plot()

.. image:: ../images/example_scan_fit.png
    :width: 400
    :alt: example fitting of multiple profiles to a 1D peak

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   Code