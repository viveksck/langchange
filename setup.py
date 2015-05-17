from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    name = "Goole N-Gram Downloader",
    ext_modules = cythonize(["googlengram/*.pyx", "googlengram/statpullscripts/*.pyx"]),
    include_dirs = [numpy.get_include()]
)
