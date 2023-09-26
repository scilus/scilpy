import os

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

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


def get_extensions():
    uncompress = Extension('scilpy.tractanalysis.uncompress',
                           ['scilpy/tractanalysis/uncompress.pyx'])
    quick_tools = Extension('scilpy.tractanalysis.quick_tools',
                            ['scilpy/tractanalysis/quick_tools.pyx'])
    grid_intersections = Extension('scilpy.tractanalysis.grid_intersections',
                                   ['scilpy/tractanalysis/grid_intersections.pyx'])
    streamlines_metrics = Extension('scilpy.tractanalysis.streamlines_metrics',
                                    ['scilpy/tractanalysis/streamlines_metrics.pyx'])
    return [uncompress, quick_tools, grid_intersections, streamlines_metrics]


class CustomBuildExtCommand(build_ext):
    """ build_ext command to use when numpy headers are needed. """

    def run(self):
        # Now that the requirements are installed, get everything from numpy
        from Cython.Build import cythonize
        from numpy import get_include

        # Add everything requires for build
        self.swig_opts = None
        self.include_dirs = [get_include()]
        self.distribution.ext_modules[:] = cythonize(
            self.distribution.ext_modules)

        # Call original build_ext command
        build_ext.finalize_options(self)
        build_ext.run(self)


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
            packages=find_packages(),
            cmdclass={'build_ext': CustomBuildExtCommand},
            ext_modules=get_extensions(),
            python_requires=PYTHON_VERSION,
            setup_requires=['cython', 'numpy', 'packaging'],
            install_requires=external_dependencies,
            entry_points={
                'console_scripts': ["{}=scripts.{}:main".format(
                    os.path.basename(s),
                    os.path.basename(s).split(".")[0]) for s in SCRIPTS]
            },
            data_files=[('data/LUT',
                         ["data/LUT/freesurfer_desikan_killiany.json",
                          "data/LUT/freesurfer_subcortical.json",
                          "data/LUT/dk_aggregate_structures.json"])],
            include_package_data=True)

setup(**opts)
