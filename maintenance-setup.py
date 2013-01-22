"""
Use this setup file to generate the C code.

$ python maintenance-setup.py build_ext --inplace
"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

my_ext_name = 'pyfelscore'

my_ext = Extension(my_ext_name, [my_ext_name + '.pyx'])

setup(
        name = my_ext_name,
        version = '0.1',
        cmdclass = { 'build_ext' : build_ext },
        ext_modules = [my_ext],
        )
