# -*- coding: utf-8 -*-

import hashlib
import inspect
import logging
import os
import pathlib
import zipfile

import requests

from scilpy import SCILPY_HOME

DVC_URL = "https://scil.usherbrooke.ca/scil_test_data/dvc-store/files/md5"


def download_file_from_google_drive(url, destination):
    """
    Download large file from Google Drive.
    Parameters
    ----------
    id: str
        id of file to be downloaded
    destination: str
        path to destination file with its name and extension
    """
    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                f.write(chunk)

    session = requests.Session()
    response = session.get(url, stream=True)

    save_response_content(response, destination)


def get_testing_files_dict():
    """ Get dictionary linking zip file to their GDrive ID & MD5SUM """
    return {
        "commit_amico.zip": "c190e6b9d22350b51e222c60febe13b4",
        "bundles.zip": "54b6e2bf2dda579886efe4e2a8989486",
        "stats.zip": "2aeac4da5ab054b3a460fc5fdc5e4243",
        "bst.zip": "eed227fd246255e7417f92d49eb1066a",
        "filtering.zip": "19116ff4244d057c8214ee3fe8e05f71",
        "ihMT.zip": "08fcf44848ba2649aad5a5a470b3cb06",
        "tractometry.zip": "890bfa70e44b15c0d044085de54e00c6",
        "bids_json.zip": "97fd9a414849567fbfdfdb0ef400488b",
        "MT.zip": "1f4345485248683b3652c97f2630950e",
        "btensor_testdata.zip": "7ada72201a767292d56634e0a7bbd9ad",
        "tracking.zip": "4793a470812318ce15f1624e24750e4d",
        "atlas.zip": "dc34e073fc582476504b3caf127e53ef",
        "anatomical_filtering.zip": "5282020575bd485e15d3251257b97e01",
        "connectivity.zip": "fe8c47f444d33067f292508d7050acc4",
        "plot.zip": "a1dc54cad7e1d17e55228c2518a1b34e",
        "others.zip": "82248b4888a63b0aeffc8070cc206995",
        "fodf_filtering.zip": "5985c0644321ecf81fd694fb91e2c898",
        "processing.zip": "eece5cdbf437b8e4b5cb89c797872e28",
        "surface_vtk_fib.zip": "241f3afd6344c967d7176b43e4a99a41",
        "tractograms.zip": "5497d0bf3ccc35f8f4f117829d790267"
    }


def fetch_data(files_dict, keys=None):
    """
    Fetch data. Typical use would be with gdown.
    But with too many data accesses, downloaded become denied.
    Using trick from https://github.com/wkentaro/gdown/issues/43.
    """

    if not os.path.exists(SCILPY_HOME):
        os.makedirs(SCILPY_HOME)

    if keys is None:
        keys = files_dict.keys()
    elif isinstance(keys, str):
        keys = [keys]
    for f in keys:
        url_md5 = files_dict[f]
        full_path = os.path.join(SCILPY_HOME, f)
        full_path_no_ext, ext = os.path.splitext(full_path)

        CURR_URL = DVC_URL + "/" + url_md5[:2] + "/" + url_md5[2:]
        if not os.path.isdir(full_path_no_ext):
            if ext == '.zip' and not os.path.isdir(full_path_no_ext):
                logging.warning('Downloading and extracting {} from url {} to '
                                '{}'.format(f, CURR_URL, SCILPY_HOME))

                # Robust method to Virus/Size check from GDrive
                download_file_from_google_drive(CURR_URL, full_path)

                with open(full_path, 'rb') as file_to_check:
                    data = file_to_check.read()
                    md5_returned = hashlib.md5(data).hexdigest()
                if md5_returned != url_md5:
                    try:
                        zipfile.ZipFile(full_path)
                    except zipfile.BadZipFile:
                        raise RuntimeError("Could not fetch valid archive for "
                                           "file {}".format(f))
                    raise ValueError('MD5 mismatch for file {}.'.format(f))

                try:
                    # If there is a root dir, we want to skip one level.
                    z = zipfile.ZipFile(full_path)
                    zipinfos = z.infolist()
                    root_dir = pathlib.Path(
                        zipinfos[0].filename).parts[0] + '/'
                    assert all([s.startswith(root_dir) for s in z.namelist()])
                    nb_root = len(root_dir)
                    for zipinfo in zipinfos:
                        zipinfo.filename = zipinfo.filename[nb_root:]
                        if zipinfo.filename != '':
                            z.extract(zipinfo, path=full_path_no_ext)
                except AssertionError:
                    # Not root dir. Extracting directly.
                    z.extractall(full_path)
            else:
                raise NotImplementedError("Data fetcher was expecting to deal "
                                          "with a zip file.")

        else:
            # toDo. Verify that data on disk is the right one.
            logging.warning("Not fetching data; already on disk.")


def get_synb0_template_path():
    """
    Return MNI 2.5mm template in scilpy repository
    Returns
    -------
    path: str
        Template path
    """
    import scilpy  # ToDo. Is this the only way?
    module_path = inspect.getfile(scilpy)
    module_path = os.path.dirname(os.path.dirname(module_path))

    path = os.path.join(module_path, 'data/',
                        'mni_icbm152_t1_tal_nlin_asym_09c_masked_2_5.nii.gz')
    return path
