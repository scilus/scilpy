# -*- coding: utf-8 -*-


import os

import yaml
from dvc import api, config

from scilpy import SCILPY_HOME, SCILPY_ROOT

DVC_REPOSITORY = "https://github.com/scilus/neurogister.git"
DEFAULT_CACHE_CONFIG = {
    "type": "symlink",
    "shared": "group",
    "dir": os.path.join(SCILPY_HOME, "dvc-cache")
}


def get_default_config():
    return config.Config(config={
        "cache": DEFAULT_CACHE_CONFIG
    })


def pull_test_case_package(package_name):
    """
    Pull a package for a test case located in the `scilpy_tests` vendor. Refer
    to the `Scilpy Tests Vendor`_ for available packages.

    Parameters
    ----------
    package_name : str
        Name of the package to pull from the `scilpy_tests` vendor.

    Returns
    -------
    str
        The path to the pulled package.

    .. _Scilpy Tests Vendor:
        https://github.com/scilus/neurogister/tree/main/store/scilpy_tests/meta.yml # noqa
    """
    with open(f"{SCILPY_ROOT}/.dvc/test_descriptors.yml", 'r') as f:
        test_descriptors = yaml.safe_load(f)

        if package_name not in test_descriptors:
            raise ValueError(f"Unknown test package: {package_name}")

        pull_package_from_dvc_repository(
            "scilpy_tests", f"{package_name}", f"{SCILPY_HOME}/test_data",
            git_revision=test_descriptors[package_name]["revision"])

        return f"{SCILPY_HOME}/test_data/{package_name}"


def pull_package_from_dvc_repository(vendor, package_name, local_destination,
                                     git_remote_url=DVC_REPOSITORY,
                                     git_revision="main",
                                     dvc_remote_name="scil-data",
                                     dvc_config_root=f"{SCILPY_ROOT}/.dvc"):
    """
    Pull a package from a correctly configured DVC remote repository. Packages
    are located in the store directory, organized in vendors with different
    purposes. Refer to the `SCIL Data Store`_ for more information.

    Parameters
    ----------
    vendor : str
        Name of the vendor to pull the package from.
    package_name : str
        Name of the package to pull from the vendor.
    local_destination : str
        Destination where to pull the package.
    git_remote_url : str
        URL of the Git repository containing DVC artifacts.
    git_revision : str
        Git revision of the DVC artifacts repository.
    dvc_remote_name : str
        Name of the DVC remote to use.
    dvc_config_root : str
        Location of the DVC configuration files.

    Returns
    -------
    str
        The path to the pulled package.

    .. _SCIL Data Store:
        https://github.com/scilus/neurogister/tree/main/store
    """
    return pull_from_dvc_repository(f"store/{vendor}/{package_name}",
                                    f"{local_destination}/",
                                    git_remote_url, git_revision,
                                    dvc_remote_name, dvc_config_root)


def pull_from_dvc_repository(remote_location, local_destination,
                             git_remote_url=DVC_REPOSITORY,
                             git_revision="main",
                             dvc_remote_name="scil-data",
                             dvc_config_root=f"{SCILPY_ROOT}/.dvc"):
    """
    Pull data from a DVC remote repository to the specified location.

    Parameters
    ----------
    remote_location : str
        Location of the data to pull in the repository.
    local_destination : str
        Destination where to pull the data.
    git_remote_url : str
        URL of the Git repository containing DVC artifacts.
    git_revision : str
        Git revision of the DVC artifacts repository.
    dvc_remote_name : str
        Name of the DVC remote to use.
    dvc_config_root : str
        Location of the DVC configuration files.

    Returns
    -------
    str
        The path to the pulled package.
    """
    repository_config = get_default_config()
    repository_config.merge(config.Config(dvc_config_root))
    remote_config = repository_config["remote"][dvc_remote_name]

    registry = api.DVCFileSystem(git_remote_url, git_revision,
                                 config=repository_config,
                                 remote_name=dvc_remote_name,
                                 remote_config=remote_config)

    registry.get(remote_location, local_destination, True, cache=True)
