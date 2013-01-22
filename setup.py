"""
This setup.py assumes that the C code has already been generated using Cython.
"""

from distutils.core import setup
from distutils.extension import Extension

my_ext_name = 'pyfelscore'

my_ext = Extension(my_ext_name, [my_ext_name + '.c'])

setup(
        name = my_ext_name,
        version = '0.1',
        ext_modules = [my_ext],
        )

