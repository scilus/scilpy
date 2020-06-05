# -*- coding: utf-8 -*-

import hashlib
import os
import logging
import shutil
import zipfile

from google_drive_downloader import GoogleDriveDownloader as gdd


# Set a user-writeable file-system location to put files:
def get_home():
    if 'SCILPY_HOME' in os.environ:
        scilpy_home = os.environ['SCILPY_HOME']
    else:
        scilpy_home = os.path.join(os.path.expanduser('~'), '.scilpy')
    return scilpy_home


def get_testing_files_dict():
    return {'atlas.zip':
            ['1UbB5rCA074LpTWxjpyhT1JBE5BjpSdUl',
             '2ba8fec24611b817d860c414463a98ee'],
            'bst.zip':
            ['1et0Kth9NHbYr-H4VeReNTYe3mx2Nq625',
             'aa86f036504f663b6bc2256eff55fd3a'],
            'bundles.zip':
            ['1LTiiTadXW9nuQt60XHIUcMpwtdnlmPvr',
             '7c7963a0965d72cff28d144ccda0fbf8'],
            'connectivity.zip':
            ['1cz6sc_9xqnEgyv5EzJ4FGhQej3GX3XLT',
             'b7795fe362b71a861e6bb1144a171f52'],
            'filtering.zip':
            ['14n9Ay7ILl9adoXdLzWYP0mpZGIaddVqg',
             'b3492123061a485e8bc87b07f25ebc3b'],
            'others.zip':
            ['11k7IZFcuehEaEsPnXG3KcClDW4FIgeVj',
             '5b21a05807265ae61d01f95f7f600992'],
            'processing.zip':
            ['1aqGNEMQReY4W5vru4ZdOiVu9ySuTLjrU',
             '25fc1806eaab5980848a8f17de992521'],
            'surface_vtk_fib.zip':
            ['1JtnHZ5XYvPcYnWOdai8YCnI-IUO3oyMp',
             '839a54adadeb1977edb2a41bc8c88c11'],
            'tracking.zip':
            ['1rDTD3f05rUxWY5Wz_UWNCjgwajHzKrg2',
             'd40e054bab9380233f3776f092ae0b0b'],
            'tractometry.zip':
            ['1-GtunE2y_z83p3YangCBCje_KP1nT9IH',
             'c99f4617dcbdea5f7a666b33682218de']}


def _get_file_md5(filename):
    """ Compute the md5 checksum of a file """
    md5_data = hashlib.md5()
    with open(filename, 'rb') as f:
        for chunk in iter(lambda: f.read(128 * md5_data.block_size), b''):
            md5_data.update(chunk)
    return md5_data.hexdigest()


def check_md5(filename, stored_md5=None):
    """
    Computes the md5 of filename and check if it matches with the supplied
    string md5

    Parameters
    -----------
    filename : string
        Path to a file.
    md5 : string
        Known md5 of filename to check against.
        If None (default), checking is skipped
    """
    if stored_md5 is not None:
        computed_md5 = _get_file_md5(filename)
        print(computed_md5)
        if stored_md5 != computed_md5:
            raise ValueError(
                '{} does not have the expected md5'.format(filename))


def _unzip(zip_file, folder):
    """ Extract the content of a zip file into a specific folder """
    z = zipfile.ZipFile(zip_file, 'r')
    z.extractall(folder)
    z.close()
    logging.info('Files successfully extracted')


def fetch_data(files_dict):
    """ Downloads files to folder and checks their md5 checksums

    Parameters
    ----------
    files_dict : dictionary
        For each file in `files_dict` the value should be (url, md5).
        The file will be downloaded from url, if the file does not already
        exist or if the file exists but the md5 checksum does not match.

    Raises
    ------
    ValueError
        Raises if the md5 checksum of the file does not match the expected
        value. The downloaded file is not deleted when this error is raised.
    """
    scilpy_home = get_home()

    if not os.path.exists(scilpy_home):
        os.makedirs(scilpy_home)

    to_dezip = {}
    for f in files_dict:
        to_dezip[f] = False
        url, md5 = files_dict[f]
        full_path = os.path.join(scilpy_home, f)

        # Zip file already exists and has the right md5sum
        if os.path.exists(full_path) and (_get_file_md5(full_path) == md5):
            continue

        # If we re-download, we re-extract
        to_dezip[f] = True
        logging.info('Downloading {} to {}'.format(f, scilpy_home))
        gdd.download_file_from_google_drive(file_id=url,
                                            dest_path=full_path,
                                            unzip=False)
        check_md5(full_path, md5)

    for f in files_dict:
        target_zip = os.path.join(scilpy_home, f)
        target_dir = os.path.splitext(os.path.join(scilpy_home,
                                                   os.path.basename(f)))[0]

        if os.path.isdir(target_dir):
            if to_dezip[f]:
                shutil.rmtree(target_dir)
                _unzip(target_zip, scilpy_home)
            else:
                logging.info('{} already extracted'.format(target_zip))
        else:
            _unzip(target_zip, scilpy_home)
            logging.info('{} successfully extracted'.format(target_zip))
