import numpy as np
from dipy.utils.optpkg import optional_package

IMPORT_ERROR_MSG = "PyTorch 2.1.2 is required to run this script. Please " + \
                   "install it first. See the official website for more " + \
                   "info: " + \
                   "https://pytorch.org/get-started/locally/"  # noqa
torch, have_torch, _ = optional_package('torch', trip_msg=IMPORT_ERROR_MSG)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def to_numpy(tensor: torch.Tensor, dtype=np.float32) -> np.ndarray:
    """ Helper function to convert a torch GPU tensor
    to numpy.
    """

    return tensor.cpu().numpy().astype(dtype)
