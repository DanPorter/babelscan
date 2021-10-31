*****
Usage
*****

**Python script**

.. code-block:: python

    import babelscan
    scan1 = babelscan.file_loader('12345.nxs')
    scan2 = babelscan.file_loader('i16_1234.dat')
    exp = babelscan.FolderMonitor('/path/to/files')
    scan3 = exp.scan(0)  # returns latest scan in directory
    scans = scan1 + scan2 + scan3  # creates MultiScan object that combines the 3 datasets


**Examples**

.. code-block:: python

    print(scan)  # prints scan information

    x = scan('axes')  # finds default xaxis in Nexus files
    y = scan('signal')  # finds default yaxis in Nexus files

    title = scan.title()  # generates a plot title

    im = scan.image(0)  # returns first detector image if scan contains 3D data
