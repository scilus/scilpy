import os

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize
from numpy import get_include


define_macros = [('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
# Modules to be compiled and include_dirs when necessary
extensions = [
    Extension('scilpy.tractograms.uncompress',
              ['src/scilpy/tractograms/uncompress.pyx'],
              define_macros=define_macros),
    Extension('scilpy.tractanalysis.voxel_boundary_intersection',
              ['src/scilpy/tractanalysis/voxel_boundary_intersection.pyx'],
              define_macros=define_macros),
    Extension('scilpy.tractanalysis.streamlines_metrics',
              ['src/scilpy/tractanalysis/streamlines_metrics.pyx'],
              define_macros=define_macros)
]

# This is the function that is executed
setup(
    name='scilpy',  # Required

    # A list of compiler Directives is available at
    # https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html#compiler-directives

    # external to be compiled
    ext_modules = cythonize(extensions, compiler_directives={"language_level": 3, "profile": False})
)