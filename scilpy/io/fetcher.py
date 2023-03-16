# -*- coding: utf-8 -*-
import logging
import hashlib
import os
import pathlib
import requests
import zipfile

GOOGLE_URL = "https://drive.google.com/uc?"

def download_file_from_google_drive(id, destination):
    """
    Download large file from Google Drive.
    Parameters
    ----------
    id: str
        id of file to be downloaded
    destination: str
        path to destination file with its name and extension
    """
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                f.write(chunk)

    session = requests.Session()
    params = {'id': id, 'confirm': True}
    response = session.get(GOOGLE_URL, params=params, stream=True)
    token = get_confirm_token(response)

    if token:
        params['confirm'] = token
        response = session.get(GOOGLE_URL, params=params, stream=True)

    save_response_content(response, destination)


def get_home():
    """ Set a user-writeable file-system location to put files. """
    if 'SCILPY_HOME' in os.environ:
        scilpy_home = os.environ['SCILPY_HOME']
    else:
        scilpy_home = os.path.join(os.path.expanduser('~'), '.scilpy')
    return scilpy_home


def get_testing_files_dict():
    """ Get dictionary linking zip file to their GDrive ID & MD5SUM """
    return {'bids_json.zip':
            ['1bMl5YtEufoKh-gjen940QTO5BpT5Y9TF',
             '521eed4911c456cc10cc3cb1f6a5dc83'],
            'plot.zip':
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
             '03aed629dea754bbc2041e7ab5f94112'],
            'anatomical_filtering.zip':
            ['1Li8DdySnMnO9Gich4pilhXisjkjz1-Dy',
             '6f0eff5154ff0973a3dc26db00e383ea'],
            'btensor_testdata.zip':
            ['1AMsKlbOZyPnT9TAbxcFzHS1b29aJWKDg',
             '7c68524fca01268203dc8bfee340f037'],
            'fodf_filtering.zip':
            ['1iyoX2ltLOoLer-v-49LHOzopHCFZ_Tv6',
             'e79c4291af584fdb25814aa7b403a6ce']}


def fetch_data(files_dict, keys=None):
    """
    Fetch data. Typical use would be with gdown.
    But with too many data accesses, downloaded become denied.
    Using trick from https://github.com/wkentaro/gdown/issues/43.
    """
    scilpy_home = get_home()

    if not os.path.exists(scilpy_home):
        os.makedirs(scilpy_home)

    if keys is None:
        keys = files_dict.keys()
    elif isinstance(keys, str):
        keys = [keys]
    for f in keys:
        url_id, md5 = files_dict[f]
        full_path = os.path.join(scilpy_home, f)
        full_path_no_ext, ext = os.path.splitext(full_path)

        CURR_URL = GOOGLE_URL + 'id=' + url_id
        if not os.path.isdir(full_path_no_ext):
            if ext == '.zip' and not os.path.isdir(full_path_no_ext):
                logging.warning('Downloading and extracting {} from url {} to '
                                '{}'.format(f, CURR_URL, scilpy_home))

                # Robust method to Virus/Size check from GDrive
                download_file_from_google_drive(url_id, full_path)

                with open(full_path, 'rb') as file_to_check:
                    data = file_to_check.read()
                    md5_returned = hashlib.md5(data).hexdigest()
                if md5_returned != md5:
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
            logging.warning("Not fetching data; already on disk.")
