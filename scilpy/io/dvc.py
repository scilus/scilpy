# -*- coding: utf-8 -*-


from dvc import config, api
import os
import yaml

from scilpy import SCILPY_HOME, SCILPY_ROOT


DVC_REPOSITORY = "https://github.com/AlexVCaron/scil_data.git"


def pull_test_case_package(package_name):
    with open(f"{SCILPY_ROOT}/.dvc/test_descriptors.yml", 'r') as f:
        test_descriptors = yaml.safe_load(f)

        if package_name not in test_descriptors:
            raise ValueError(f"Unknown test case: {package_name}")

        pull_package_from_dvc_repository(
            package_name, f"{SCILPY_HOME}/test_data",
            test_descriptors[package_name]["revision"])

        return f"{SCILPY_HOME}/test_data/{package_name}"


def pull_package_from_dvc_repository(package_name, output_dir, revision="main",
                                     remote_url=DVC_REPOSITORY,
                                     remote_name="scil-data",
                                     dvc_config_root=f"{SCILPY_ROOT}/.dvc"):
    """
    Pull a package from a correctly configured DVC remote repository. The
    trivial valid configuration is a data storage located at the registry
    root and named store.
    """
    return pull_from_dvc_repository(f"store/{package_name}",
                                    f"{output_dir}/{package_name}",
                                    revision, remote_url, remote_name,
                                    dvc_config_root)


def pull_from_dvc_repository(package_name, output_dir, revision="main",
                             remote_url=DVC_REPOSITORY,
                             remote_name="scil-data",
                             dvc_config_root=f"{SCILPY_ROOT}/.dvc"):
    """
    Pull data from a DVC remote repository. This is mostly used in
    conjuction with SCILPY data DVC endpoints at :
        - https://github.com/AlexVCaron/scil_data.git
        - https://scil.usherbrooke.ca/scil_test_data/dvc-store
    """

    conf = config.Config(dvc_config_root)
    registry = api.DVCFileSystem(remote_url, revision,
                                 remote_name=remote_name,
                                 remote_config=conf["remote"][remote_name])

    registry.get(package_name, output_dir, True)
