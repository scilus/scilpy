import os

from Cython.Build import cythonize
import numpy
from setuptools import setup, find_packages
from setuptools.extension import Extension
PACKAGES = find_packages()

# Get version and release info, which is all stored in scilpy/version.py
ver_file = os.path.join('scilpy', 'version.py')
with open(ver_file) as f:
    exec(f.read())
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
            packages=PACKAGES,
            install_requires=REQUIRES,
            requires=REQUIRES,
            scripts=SCRIPTS)

extensions = [Extension('scilpy.tractanalysis.streamlines_metrics',
                        ['scilpy/tractanalysis/streamlines_metrics.pyx'],
                        include_dirs=[numpy.get_include()])]

opts['ext_modules'] = cythonize(extensions)


if __name__ == '__main__':
    setup(**opts)
