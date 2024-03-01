# -*- coding: utf-8 -*-


from dvc import config, api
import os
import yaml

from scilpy import SCILPY_HOME, SCILPY_ROOT


DVC_REPOSITORY = "https://github.com/AlexVCaron/scil_data.git"


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
        https://github.com/AlexVCaron/scil_data/tree/main/store/scilpy_tests
    """
    with open(f"{SCILPY_ROOT}/.dvc/test_descriptors.yml", 'r') as f:
        test_descriptors = yaml.safe_load(f)

        if package_name not in test_descriptors:
            raise ValueError(f"Unknown test package: {package_name}")

        pull_package_from_dvc_repository(
            "scilpy_tests", f"{package_name}", f"{SCILPY_HOME}/test_data",
            test_descriptors[package_name]["revision"])

        return f"{SCILPY_HOME}/test_data/{package_name}"


def pull_package_from_dvc_repository(vendor, package_name, output_dir,
                                     revision="main",
                                     remote_url=DVC_REPOSITORY,
                                     remote_name="scil-data",
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
    output_dir : str
        Path to the directory where the package will be pulled.
    revision : str
        Git revision of the DVC artifact to pull.
    remote_url : str
        URL of the DVC git repository.
    remote_name : str
        Name of the DVC remote to use.
    dvc_config_root : str
        Location of the DVC configuration files.

    Returns
    -------
    str
        The path to the pulled package.

    .. _SCIL Data Store:
        https://github.com/AlexVCaron/scil_data/tree/main/store
    """
    return pull_from_dvc_repository(f"store/{vendor}/{package_name}",
                                    f"{output_dir}/{package_name}",
                                    revision, remote_url, remote_name,
                                    dvc_config_root)


def pull_from_dvc_repository(package_name, output_dir, revision="main",
                             remote_url=DVC_REPOSITORY,
                             remote_name="scil-data",
                             dvc_config_root=f"{SCILPY_ROOT}/.dvc"):
    """
    Pull data from a DVC remote repository to the specified location.

    Parameters
    ----------
    package_name : str
        Name of the package to pull from the DVC repository.
    output_dir : str
        Path to the directory where the package will be pulled.
    revision : str
        Git revision of the DVC artifact to pull.
    remote_url : str
        URL of the DVC git repository.
    remote_name : str
        Name of the DVC remote to use.
    dvc_config_root : str
        Location of the DVC configuration files.

    Returns
    -------
    str
        The path to the pulled package.
    """
    conf = config.Config(dvc_config_root)
    registry = api.DVCFileSystem(remote_url, revision,
                                 remote_name=remote_name,
                                 remote_config=conf["remote"][remote_name])

    registry.get(package_name, output_dir, True)
