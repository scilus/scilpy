# -*- coding: utf-8 -*-
import numpy as np
import pyopencl as cl
from pyopencl import mem_flags as mf


FLAT_INDEX_CL_CODE = """
int get_flat_index(const int x, const int y,
                   const int z, const int w,
                   const int xLen,
                   const int yLen,
                   const int zLen)
{{
    return x +
           y * xLen +
           z * (xLen)
             * (yLen) +
           w * (yLen)
             * (yLen)
             * (zLen);
}}
"""


class CLManager(object):
    class OutBuffer(object):
        def __init__(self, buf, shape, dtype):
            self.buf = buf
            self.shape = shape
            self.dtype = dtype

    def __init__(self):
        self.context = cl.create_some_context(interactive=False)
        self.queue = cl.CommandQueue(self.context)
        self.input_buffers = []
        self.output_buffers = []

    def add_input_buffer(self, arr, dtype=np.float32):
        # convert to fortran ordered, float32 array
        arr = np.asfortranarray(arr, dtype=dtype)
        buf = cl.Buffer(self.context,
                        mf.READ_ONLY | mf.COPY_HOST_PTR,
                        hostbuf=arr)
        self.input_buffers.append(buf)

    def add_output_buffer(self, shape, dtype):
        buf = cl.Buffer(self.context, mf.WRITE_ONLY,
                        np.prod(shape) * np.dtype(dtype).itemsize)
        self.output_buffers.append(self.OutBuffer(buf, shape, dtype))

    def add_program(self, code_str, kernel_name):
        program = cl.Program(self.context, code_str).build()
        self.kernel = cl.Kernel(program, kernel_name)

    def run(self, global_size, local_size=None):
        wait_event = self.kernel(self.queue,
                                 global_size,
                                 local_size,
                                 *self.input_buffers,
                                 *[out.buf for out in self.output_buffers])
        outputs = []
        for output in self.output_buffers:
            out_arr = np.empty(output.shape, dtype=output.dtype, order='F')
            cl.enqueue_copy(self.queue, out_arr, output.buf,
                            wait_for=[wait_event])
            outputs.append(out_arr)
        return outputs


class CLKernel(object):
    class KernelConstVar(object):
        def __init__(self, ctype, value):
            self.ctype = ctype
            self.value = value

    def __init__(self):
        self.code = ""
        self.entrypoint = ""
        self.constants = {}

    def add_constant(self, var_name, ctype, value):
        var_name = var_name.capitalize()
        if var_name in self.constants:
            raise ValueError('Constant {0} already defined in kernel.'
                             .format(var_name))
        self.constants[var_name] = self.KernelConstVar(ctype, value)

    def set_kernel_code(self, code_str, entrypoint):
        self.code = code_str
        self.entrypoint = entrypoint

    def __str__(self):
        code_str = ""

        # write constant values
        for cname in self.constants:
            const_var = self.constants[cname]
            code_str += "__constant {0} {1} = {2};\n".format(const_var.ctype,
                                                             cname,
                                                             const_var.value)

        # add helper functions
        code_str += FLAT_INDEX_CL_CODE + "\n"

        # write actual kernel code
        code_str += self.code
        return code_str


def angle_aware_bilateral_filtering_cl():
    """
    1.  convert arrays to float32
    2.  convert arrays to fortran ordering
    3.  generate buffers
    4.  kernel code
    4.1 kernel code includes range filtering
    4.2 kernel code processes all sphere directions for a voxel
    4.3 conversion to full basis is done on the GPU
    5.  return output in full SH basis

    OPTIM: Do not pad, return 0 when outside
           image dimensions directly in kernel.
    """
    pass
