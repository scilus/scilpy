# -*- coding: utf-8 -*-
import numpy as np
import logging
import inspect
import os
import scilpy

from dipy.utils.optpkg import optional_package
cl, have_opencl, _ = optional_package('pyopencl')


def cl_device_type(device_type_str):
    if device_type_str == 'cpu':
        return cl.device_type.CPU
    if device_type_str == 'gpu':
        return cl.device_type.GPU
    return -1


class CLManager(object):
    """
    Class for managing an OpenCL program.

    Wraps a subset of pyopencl functions to simplify its
    integration with python. The OpenCL program can be run
    on the cpu or on the gpu, given the appropriate drivers
    are installed.

    When multiple cpu or gpu are available, the
    one that first comes up in the list of available devices
    is selected.

    Parameters
    ----------
    cl_kernel: CLKernel object
        The CLKernel containing the OpenCL program to manage.
    device_type: string
        The device onto which to run the program. One of 'cpu', 'gpu'.
    """
    def __init__(self, cl_kernel, device_type='gpu'):
        if not have_opencl:
            raise RuntimeError('pyopencl is not installed. '
                               'Cannot create CLManager instance.')

        # Reduce verbose level for pyopencl
        logging.getLogger('pytools.persistent_dict').setLevel(logging.CRITICAL)
        logging.getLogger('pyopencl').setLevel(logging.CRITICAL)

        self.input_buffers = []
        self.output_buffers = []

        # maps key to index in buffers list
        self.inputs_mapping = {}
        self.outputs_mapping = {}

        # Find the best device for running GPU tasks
        platforms = cl.get_platforms()
        best_device = None
        for p in platforms:
            devices = p.get_devices()
            for d in devices:
                d_type = d.get_info(cl.device_info.TYPE)
                if d_type == cl_device_type(device_type)\
                   and best_device is None:
                    best_device = d  # take the first device of right type

        if best_device is None:
            raise ValueError('No device of type {} found'.format(device_type))

        self.context = cl.Context(devices=[best_device])
        self.queue = cl.CommandQueue(self.context)
        program = cl.Program(self.context, cl_kernel.code_string).build()
        self.kernel = cl.Kernel(program, cl_kernel.entry_point)

    class OutBuffer(object):
        """
        Structure containing output buffer information.

        Parameters
        ----------
        buf: cl.Buffer
            The cl.Buffer object containing the output.
        shape: tuple
            Shape for the output array.
        dtype: dtype
            Datatype for output.
        """
        def __init__(self, buf, shape, dtype):
            self.buf = buf
            self.shape = shape
            self.dtype = dtype

    def add_input_buffer(self, key, arr=None, dtype=np.float32):
        """
        Add an input buffer to the kernel program. Input buffers
        must be added in the same order as they are declared inside
        the kernel code (.cl file).

        Parameters
        ----------
        key: string
            Name of the buffer in the input buffers list. Used for
            referencing when updating buffers.
        arr: numpy ndarray
            Data array.
        dtype: dtype, optional
            Optional type for array data. It is recommended to use float32
            whenever possible to avoid unexpected behaviours.

        Note
        ----
        Array is reordered as fortran array and then flattened. This is
        important to keep in mind when writing kernel code.

        For example, for a 3-dimensional array of shape (X, Y, Z), the flat
        index for position i, j, k is idx = i + j * X + z * X * Y.
        """
        buf = None
        if arr is not None:
            # convert to fortran ordered, dtype array
            arr = np.asfortranarray(arr, dtype=dtype)
            buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY |
                            cl.mem_flags.COPY_HOST_PTR, hostbuf=arr)

        if key in self.inputs_mapping.keys():
            raise ValueError('Invalid key for buffer!')

        self.inputs_mapping[key] = len(self.input_buffers)
        self.input_buffers.append(buf)

    def update_input_buffer(self, key, arr, dtype=np.float32):
        """
        Update an input buffer. Input buffers must first be added
        to program using `add_input_buffer`.

        Parameters
        ----------
        key: string
            Name of the buffer in the input buffers list.
        arr: numpy ndarray
            Data array.
        dtype: dtype, optional
            Optional type for array data. It is recommended to use float32
            whenever possible to avoid unexpected behaviours.
        """
        if key not in self.inputs_mapping.keys():
            raise ValueError('Invalid key for buffer!')
        argpos = self.inputs_mapping[key]

        arr = np.asfortranarray(arr, dtype=dtype)
        buf = cl.Buffer(self.context,
                        cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                        hostbuf=arr)
        self.input_buffers[argpos] = buf

    def add_output_buffer(self, key, shape=None, dtype=np.float32):
        """
        Add an output buffer to the kernel program. Output buffers
        must be added in the same order as they are declared inside
        the kernel code (.cl file).

        Parameters
        ----------
        key: string
            Name of the buffer in the output buffers list. Used for
            referencing when updating buffers.
        shape: tuple
            Shape of the output array.
        dtype: dtype, optional
            Optional type for array data. It is recommended to use float32
            whenever possible to avoid unexpected behaviours.
        """
        if key in self.outputs_mapping.keys():
            raise ValueError('Invalid key for buffer!')

        buf = None
        if shape is not None:
            buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY,
                            np.prod(shape) * np.dtype(dtype).itemsize)

        self.outputs_mapping[key] = len(self.output_buffers)
        self.output_buffers.append(self.OutBuffer(buf, shape, dtype))

    def update_output_buffer(self, key, shape, dtype=np.float32):
        """
        Update an output buffer. Output buffers must first be added
        to program using `add_output_buffer`.

        Parameters
        ----------
        key: string
            Name of the buffer in the output buffers list.
        shape: tuple
            New shape of the output array.
        dtype: dtype, optional
            Optional type for array data. It is recommended to use float32
            whenever possible to avoid unexpected behaviours.
        """
        if key not in self.outputs_mapping.keys():
            raise ValueError('Invalid key for buffer!')
        argpos = self.outputs_mapping[key]

        buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY,
                        np.prod(shape) * np.dtype(dtype).itemsize)
        out_buf = self.OutBuffer(buf, shape, dtype)
        self.output_buffers[argpos] = out_buf

    def run(self, global_size, local_size=None):
        """
        Execute the kernel code on the GPU.

        Parameters
        ----------
        global_size: tuple
            Tuple of between 1 and 3 entries representing the shape of the
            grid used for GPU computing. OpenCL uses global_size to generate
            a unique id for each kernel execution, which can be queried using
            get_global_id(axis) with axis between 0 and 2.
        local_size: tuple, optional
            Dimensions of local groups. Must divide global_size exactly,
            element-wise. If None, an implementation local workgroup size is
            used. Memory allocated in the __local address space on the GPU is
            shared between elements in a same workgroup.

        Returns
        -------
        outputs: list of ndarrays
            List of outputs produced by the program.
        """
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
    """
    Wrapper for OpenCL kernel/program code.

    Parameters
    ----------
    entrypoint: string
        Name of __kernel function in .cl file.
    module: string
        Scilpy module in which the kernel code is located.
    filename: string
        Name for the file containing the kernel code.
    """
    def __init__(self, entrypoint, module, filename):
        path_to_kernel = self._get_kernel_path(module, filename)
        try:
            f = open(path_to_kernel, 'r')
        except Exception:
            raise ValueError('OpenCL file not found in {}'
                             .format(path_to_kernel))
        self.code = f.readlines()
        self.entrypoint = entrypoint

    def _get_kernel_path(self, module, filename):
        """
        Get the full path for the OpenCL kernel located in scilpy
        module `module` with filename `filename`.
        """
        module_path = inspect.getfile(scilpy)
        kernel_path = os.path.join(os.path.dirname(module_path),
                                   module, filename)
        return kernel_path

    def set_define(self, def_name, value):
        """
        Set the value for a compiler definition in the kernel code.
        This method will overwrite the previous value for this definition.

        Parameters
        ----------
        def_name: string
            Name of definition. By convention, #define should be in upper case.
            Therefore, this value will also be converted to upper case.
        value: string
            The value for the define. Will be replaced directly in the kernel
            code.

        Note
        ----
        Be careful! #define instructions are not typed and therefore prone to
        compilation errors. They are however faster to access than const
        variables. Moreover, they do not take additional space on the GPU.
        """
        def_name = def_name.upper()
        to_find = '#define {}'.format(def_name)
        def_line = None
        for i, line in enumerate(self.code):
            if line.find(to_find) != -1:
                def_line = i
                break
        if def_line is None:
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
