import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scilpy.ml.utils import to_numpy  # noqa: E402


def test_to_numpy_half_precision():
    bfloat_tensor = torch.ones((3, 3), dtype=torch.bfloat16)
    arr = to_numpy(bfloat_tensor)
    assert arr.dtype == np.float32
    assert np.allclose(arr, 1.0)

    half_tensor = torch.ones((3, 3), dtype=torch.float16)
    arr_half = to_numpy(half_tensor)
    assert arr_half.dtype == np.float32
    assert np.allclose(arr_half, 1.0)


def test_to_numpy_preserves_requested_dtype_precision():
    """
    .float() must not be applied unconditionally: forcing an already
    numpy-compatible tensor (float32/float64) through float32 first would
    silently truncate precision before the requested dtype is applied.
    """
    double_tensor = torch.tensor([1.0 / 3.0], dtype=torch.float64)
    arr = to_numpy(double_tensor, dtype=np.float64)
    assert arr.dtype == np.float64
    # If the tensor had been downcast to float32 first, this would fail:
    # float32(1/3) astype float64 != true float64(1/3).
    assert arr[0] == np.float64(1.0 / 3.0)
