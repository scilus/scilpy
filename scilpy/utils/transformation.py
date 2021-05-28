# -*- coding: utf-8 -*-

from dipy.io.stateful_tractogram import StatefulTractogram
import numpy as np

def get_axis_flip_vector(flip_axes):
    flip_vector = np.ones(3)
    if 'x' in flip_axes:
        flip_vector[0] = -1.0
    if 'y' in flip_axes:
        flip_vector[1] = -1.0
    if 'z' in flip_axes:
        flip_vector[2] = -1.0

    return flip_vector


def get_shift_vector(sft):
    dims = sft.space_attributes[1]
    shift_vector = -1.0 * (np.array(dims) / 2.0)

    return shift_vector


def flip_sft(sft, flip_axes):
    flip_vector = get_axis_flip_vector(flip_axes)
    shift_vector = get_shift_vector(sft)

    flipped_streamlines = []

    streamlines = sft.streamlines

    for streamline in streamlines:
        mod_streamline = streamline + shift_vector
        mod_streamline *= flip_vector
        mod_streamline -= shift_vector
        flipped_streamlines.append(mod_streamline)

    new_sft = StatefulTractogram.from_sft(flipped_streamlines, sft,
                                          data_per_point=sft.data_per_point,
                                          data_per_streamline=sft.data_per_streamline)
    return new_sft