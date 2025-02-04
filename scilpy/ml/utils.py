from dipy.utils.optpkg import optional_package

IMPORT_ERROR_MSG = "PyTorch is required to run this script. Please install" + \
                   " it first. See the official website for more info: " + \
                   "https://pytorch.org/get-started/locally/" # noqa
torch, have_torch, _ = optional_package('pytorch', trip_msg=IMPORT_ERROR_MSG)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
