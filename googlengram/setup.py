from distutils.core import setup
from Cython.Build import cythonize

setup(
    name = "Goole N-Gram Downloader",
    ext_modules = cythonize(["./*.pyx", "./statpullscripts/*.pyx"])
)
