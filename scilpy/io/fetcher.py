# -*- coding: utf-8 -*-

import hashlib
import logging
import os
import shutil
from time import sleep
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
    """ Get dictionary linking zip file to their GDrive ID & MD5SUM """
    return {'atlas.zip':
            ['1waYx4ED3qwzyJqrICjjgGXXBW2v4ZCYJ',
             '0c1d3da231d1a8b837b5d756c9170b08'],
            'bst.zip':
            ['1YprJRnyXk7VRHUkb-bJLs69C1v3tPd1S',
             'c0551a28dcefcd7cb53f572b1794b3e8'],
            'bundles.zip':
            ['1VaGWwhVhnfsZBCCYu12dta9qi0SgZFP7',
             '5fbf5c8eaabff2648ad509e06b003e67'],
            'commit_amico.zip':
            ['1vyMtQd1u2h2pza9M0bncDWLc34_4MRPK',
             '12e901e899ee48bdf31b25f22d39ee48'],
            'connectivity.zip':
            ['1lZqiOKmwTluPIRqblthOnBc4KI2kfKUC',
             '2fed405d8241b9b5fc43e6d640008ae9'],
            'filtering.zip':
            ['1yzHSL4tBtmm_aeI1i0qJhrA9z040k0im',
             'dbe796fb75c3e1e5559fad3308982769'],
            'others.zip':
            ['12BAszPjE1A9L2RbQJIFpkPzqUJfPdYO6',
             '075deda4532042192c4103df4371ecb4'],
            'processing.zip':
            ['1caaKoAChyPs5c4WemQWUsR-efD_q2z_b',
             '57aee810f2f5c687df48de65935bb527'],
            'surface_vtk_fib.zip':
            ['1c9KMNFeSkyYDgu3SH_aMf0kduIlpt7cN',
             '946beb4271b905a2bd69ad2d80136ca9'],
            'tracking.zip':
            ['1QSekZYDoMvv-An6FRMSt_s_qPeB3BHfw',
             '78129871fbefadf5cede7b1c6a7c9cc5'],
            'tractometry.zip':
            ['130mxBo4IJWPnDFyOELSYDif1puRLGHMX',
             '32f938ae33a6c185346d2574240ebf3b']}


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
        if stored_md5 != computed_md5:
            return False
    return True


def _unzip(zip_file, folder):
    """ Extract the content of a zip file into a specific folder """
    z = zipfile.ZipFile(zip_file, 'r')
    z.extractall(folder)
    z.close()
    logging.info('Files successfully extracted')


def fetch_data(files_dict, keys=None):
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

    to_unzip = {}
    if keys is None:
        keys = files_dict.keys()
    elif isinstance(keys, str):
        keys = [keys]
    for f in keys:
        tryout = 0
        while tryout < 3:
            to_unzip[f] = False
            url, md5 = files_dict[f]
            full_path = os.path.join(scilpy_home, f)

            # Zip file already exists and has the right md5sum
            if os.path.exists(full_path) and (_get_file_md5(full_path) == md5):
                break
            elif os.path.exists(full_path):
                if tryout > 0:
                    logging.error('Wrong md5sum after {} attemps for {}'.format(
                        tryout+1, full_path))
                os.remove(full_path)

            # If we re-download, we re-extract
            to_unzip[f] = True
            logging.info('Downloading {} to {}'.format(f, scilpy_home))
            gdd.download_file_from_google_drive(file_id=url,
                                                dest_path=full_path,
                                                unzip=False)
            if check_md5(full_path, md5):
                break
            else:
                tryout += 1
                sleep(10)

    for f in keys:
        target_zip = os.path.join(scilpy_home, f)
        target_dir = os.path.splitext(os.path.join(scilpy_home,
                                                   os.path.basename(f)))[0]

        if os.path.isdir(target_dir):
            if to_unzip[f]:
                shutil.rmtree(target_dir)
                _unzip(target_zip, scilpy_home)
            else:
                logging.info('{} already extracted'.format(target_zip))
        else:
            _unzip(target_zip, scilpy_home)
            logging.info('{} successfully extracted'.format(target_zip))
