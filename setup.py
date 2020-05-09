import os
import sys

from pkg_resources import parse_requirements
from setuptools import setup, find_packages
from setuptools.extension import Extension
PACKAGES = find_packages()

try:
    from Cython.Build import cythonize
except ImportError:
    def cythonize(*args, **kwargs):
        from Cython.Build import cythonize
        return cythonize(*args, **kwargs)

try:
    from numpy import get_include
except ImportError:
    def get_include():
        from numpy import get_include
        return get_include()

with open('requirements.txt') as f:
    required_dependencies = f.read().splitlines()
    external_dependencies = []
    for dependency in required_dependencies:
        if dependency[0:2] == '-e':
            repo_name = dependency.split('=')[-1]
            repo_url = dependency[3:]
            external_dependencies.append('{} @ {}'.format(repo_name, repo_url))
        else:
            external_dependencies.append(dependency)

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
            setup_requires=['cython', 'numpy'],
            install_requires=external_dependencies,
            scripts=SCRIPTS,
            data_files=[('data/LUT',
                         ["data/LUT/freesurfer_desikan_killiany.json",
                          "data/LUT/freesurfer_subcortical.json"])],
            include_package_data=True)


if __name__ == '__main__':
    setup(**opts)
    if sys.argv[1] != 'clean':
        extensions = [Extension('scilpy.tractanalysis.uncompress',
                                ['scilpy/tractanalysis/uncompress.pyx'],
                                include_dirs=[get_include()]),
                        Extension('scilpy.tractanalysis.quick_tools',
                                ['scilpy/tractanalysis/quick_tools.pyx'],
                                include_dirs=[get_include()]),
                        Extension('scilpy.tractanalysis.grid_intersections',
                                ['scilpy/tractanalysis/grid_intersections.pyx'],
                                include_dirs=[get_include()]),
                        Extension('scilpy.tractanalysis.streamlines_metrics',
                                ['scilpy/tractanalysis/streamlines_metrics.pyx'],
                                include_dirs=[get_include()])]
        opts['ext_modules'] = cythonize(extensions)
        setup(**opts)
