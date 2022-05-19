# -*- coding: utf-8 -*-

import logging
import os

from gdown import cached_download, extractall

GOOGLE_URL = "https://drive.google.com/uc?id="


# Set a user-writeable file-system location to put files:
def get_home():
    if 'SCILPY_HOME' in os.environ:
        scilpy_home = os.environ['SCILPY_HOME']
    else:
        scilpy_home = os.path.join(os.path.expanduser('~'), '.scilpy')
    return scilpy_home


def get_testing_files_dict():
    """ Get dictionary linking zip file to their GDrive ID & MD5SUM """
    return {'plot.zip':
            ['1Ab-oVWI1Fu7fHTEz1H3-s1TfR_oW-GOE',
             'cca8f1e19da357f44365a7e27b9029ca'],
            'ihMT.zip':
            ['1V0xzvmVrVlL9dRKhc5-7xWESkmof1zyS',
             '5d28430ac46b4fc04b6d77f9efaefb5c'],
            'MT.zip':
            ['1C2LEUkGaLFdsmym3kBrAtfPjPtv5mJuZ',
             '13532c593efdf09350667df14ea4e93a'],
            'atlas.zip':
            ['1waYx4ED3qwzyJqrICjjgGXXBW2v4ZCYJ',
             'eb37427054cef5d50ac3d429ff53de47'],
            'bst.zip':
            ['1YprJRnyXk7VRHUkb-bJLs69C1v3tPd1S',
             'c0551a28dcefcd7cb53f572b1794b3e8'],
            'bundles.zip':
            ['1VaGWwhVhnfsZBCCYu12dta9qi0SgZFP7',
             '5fbf5c8eaabff2648ad509e06b003e67'],
            'commit_amico.zip':
            ['1vyMtQd1u2h2pza9M0bncDWLc34_4MRPK',
             'b40800ab4290e4f58c375140fe59b44f'],
            'connectivity.zip':
            ['1lZqiOKmwTluPIRqblthOnBc4KI2kfKUC',
             '6d13bd076225fa2f786f416fa754623a'],
            'filtering.zip':
            ['1yzHSL4tBtmm_aeI1i0qJhrA9z040k0im',
             'dbe796fb75c3e1e5559fad3308982769'],
            'others.zip':
            ['12BAszPjE1A9L2RbQJIFpkPzqUJfPdYO6',
             '981dccd8b23aad43aa014f4fdd907e70'],
            'processing.zip':
            ['1caaKoAChyPs5c4WemQWUsR-efD_q2z_b',
             'a2f982b8d84833f5ccfe709b725307d2'],
            'surface_vtk_fib.zip':
            ['1c9KMNFeSkyYDgu3SH_aMf0kduIlpt7cN',
             '946beb4271b905a2bd69ad2d80136ca9'],
            'tracking.zip':
            ['1QSekZYDoMvv-An6FRMSt_s_qPeB3BHfw',
             '6d88910403fb4d9b79604f11100d8915'],
            'tractometry.zip':
            ['130mxBo4IJWPnDFyOELSYDif1puRLGHMX',
             '3e27625a1e7f2484b7fa5028c95324cc'],
            'stats.zip':
            ['1vsM7xuU0jF5fL5PIgN6stAH7oO683tw0',
             'bcc21835cf0bf7210bdc99ba5d8df44b'],
            'anatomical_filtering.zip':
            ['1Li8DdySnMnO9Gich4pilhXisjkjz1-Dy',
             '6f0eff5154ff0973a3dc26db00e383ea']}


def fetch_data(files_dict, keys=None):
    """ Downloads files to folder and checks their md5 checksums

    Parameters
    ----------
    files_dict : dictionary
        For each file in `files_dict` the value should be (url, md5).
        The file will be downloaded from url, if the file does not already
        exist or if the file exists but the md5 checksum does not match.
    keys: List[str] or str
        Files to get. Must be keys in the files_dict. If None is given, fetch
        all the keys from the files_dict.

    Raises
    ------
    ValueError
        Raises if the md5 checksum of the file does not match the expected
        value. The downloaded file is not deleted when this error is raised.
    """
    scilpy_home = get_home()

    if not os.path.exists(scilpy_home):
        os.makedirs(scilpy_home)

    if keys is None:
        keys = files_dict.keys()
    elif isinstance(keys, str):
        keys = [keys]
    for f in keys:
        url, md5 = files_dict[f]
        full_path = os.path.join(scilpy_home, f)

        logging.info('Downloading {} to {}'.format(f, scilpy_home))
        cached_download(url=GOOGLE_URL+url,
                        path=full_path,
                        md5=md5,
                        quiet=True,
                        postprocess=extractall)
