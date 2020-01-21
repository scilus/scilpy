import os

from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.extension import Extension
PACKAGES = find_packages()

# Get version and release info, which is all stored in scilpy/version.py
ver_file = os.path.join('scilpy', 'version.py')
with open(ver_file) as f:
    exec(f.read())


class build_inplace_all_ext(build_ext):

    description = "build optimized code (.pyx files) " +\
                  "(compile/link inplace)"

    def finalize_options(self):
        # Force inplace building for ease of importation
        # self.inplace = True
        build_ext.finalize_options(self)
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())


opts = dict(name=NAME,
            maintainer=MAINTAINER,
            maintainer_email=MAINTAINER_EMAIL,
            description=DESCRIPTION,
            long_description=LONG_DESCRIPTION,
            url=URL,
            download_url=DOWNLOAD_URL,
            license=LICENSE,
            classifiers=CLASSIFIERS,
            author=AUTHOR,
            author_email=AUTHOR_EMAIL,
            platforms=PLATFORMS,
            version=VERSION,
            packages=find_packages(),
            setup_requires=["cython (>=0.29.12)", "numpy (>=1.16.2)"],
            install_requires=REQUIRES,
            requires=REQUIRES,
            scripts=SCRIPTS,
            ext_modules=[
                Extension('scilpy.tractanalysis.uncompress',
                          ['scilpy/tractanalysis/uncompress.pyx']),
                Extension('scilpy.tractanalysis.quick_tools',
                          ['scilpy/tractanalysis/quick_tools.pyx']),
                Extension('scilpy.tractanalysis.streamlines_metrics',
                          ['scilpy/tractanalysis/streamlines_metrics.pyx'])],
            cmdclass={'build_ext': build_inplace_all_ext})


if __name__ == '__main__':
    setup(**opts)
