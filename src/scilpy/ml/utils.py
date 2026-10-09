from dipy.utils.optpkg import optional_package
import numpy as np

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


def to_numpy(tensor: "torch.Tensor", dtype=np.float32) -> np.ndarray:
    """ Helper function to convert a torch GPU tensor
    to numpy.
    """
    # Detach removes gradient tracking. bfloat16/float16 have no direct
    # NumPy export on some PyTorch/CPU combinations, so upcast only those
    # to float32 first. Other dtypes convert directly, so a caller
    # requesting dtype=np.float64 does not lose precision to an
    # unconditional float32 downcast beforehand.
    tensor = tensor.detach().cpu()
    if tensor.dtype in (torch.bfloat16, torch.float16):
        tensor = tensor.float()
    return tensor.numpy().astype(dtype)
