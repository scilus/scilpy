# -*- coding: utf-8 -*-
import numpy as np
import inspect
import os
import scilpy

from dipy.utils.optpkg import optional_package
cl, have_opencl, _ = optional_package('pyopencl')


class CLManager(object):
    class OutBuffer(object):
        def __init__(self, buf, shape, dtype):
            self.buf = buf
            self.shape = shape
            self.dtype = dtype

    def __init__(self, cl_kernel):
        self.input_buffers = []
        self.output_buffers = []

        self.context = cl.create_some_context(interactive=False)
        self.queue = cl.CommandQueue(self.context)
        program = cl.Program(self.context, cl_kernel.code_string).build()
        self.kernel = cl.Kernel(program, cl_kernel.entry_point)

    def add_input_buffer(self, arr, dtype=np.float32):
        # convert to fortran ordered, float32 array
        arr = np.asfortranarray(arr, dtype=dtype)
        buf = cl.Buffer(self.context,
                        cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                        hostbuf=arr)
        self.input_buffers.append(buf)

    def add_output_buffer(self, shape, dtype):
        buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY,
                        np.prod(shape) * np.dtype(dtype).itemsize)
        self.output_buffers.append(self.OutBuffer(buf, shape, dtype))

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

    def __init__(self, entrypoint, path_to_kernel):
        f = open(path_to_kernel, 'r')
        self.code = f.readlines()
        self.entrypoint = entrypoint

    def set_define(self, def_name, value):
        # warning! #define are not typed and therefore prone to compilation
        # error. They are however faster than accessing a const variable on
        # the GPU.
        def_name = def_name.upper()
        to_find = '#define {}'.format(def_name)
        def_line = -1
        for i, line in enumerate(self.code):
            if line.find(to_find) != -1:
                if def_line != -1:
                    raise ValueError('Multiple definitions for {0}'
                                     .format(def_name))
                def_line = i
                break
        if def_line == -1:
            raise ValueError('Definition {0} not found in kernel code'
                             .format(def_name))

        self.code[def_line] = '#define {0} {1}\n'.format(def_name, value)

    @property
    def entry_point(self):
        return self.entrypoint

    @property
    def code_string(self):
        code_str = ''.join(self.code)
        return code_str


def get_kernel_path(module, kernel_name):
    module_path = inspect.getfile(scilpy)
    kernel_path = os.path.join(os.path.dirname(module_path),
                               module, kernel_name)
    return kernel_path
