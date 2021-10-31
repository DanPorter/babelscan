from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()


setup(
    name='babelscan',
    version='0.6.2',
    packages=['babelscan'],
    url='https://github.com/DanPorter/babelscan',
    license='Apache 2.0',
    author='dgporter',
    author_email='dan.porter@diamond.ac.uk',
    description='BabelScan is a format independent data structure for holding different types of data from a scan file',
    long_description_content_type='text/markdown',
    long_description=readme(),
    keywords=[
        'nexus', 'nexusformat', 'hdf', 'scan', 'data',
        'crystal', 'diffraction', 'crystallography', 'science',
        'x-ray', 'neutron'
        ],
    classifiers=[
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.7',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'Development Status :: 3 - Alpha',
        ],
    install_requires=['numpy', 'h5py', 'imageio', 'python-dateutil']
)
