#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import get_testing_files_dict, fetch_data, get_home


# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict())
tmp_dir = tempfile.TemporaryDirectory()


def test_convert_fib(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    input_fib = os.path.join(get_home(), 'surface_vtk_fib',
                             'gyri_fanning.fib')
    input_fa = os.path.join(get_home(), 'surface_vtk_fib',
                            'fa.nii.gz')
    ret = script_runner.run('scil_convert_tractogram.py', input_fib,
                            'gyri_fanning.trk', '--reference', input_fa)
    return ret.success


def test_convert_surface(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    input_surf = os.path.join(get_home(), 'surface_vtk_fib',
                              'lhpialt.vtk')
    ret = script_runner.run('scil_convert_surface.py', input_surf,
                            'rhpialt.ply')
    return ret.success


def test_compress_fib(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    input_fib = os.path.join(get_home(), 'surface_vtk_fib',
                             'gyri_fanning.fib')
    input_fa = os.path.join(get_home(), 'surface_vtk_fib',
                            'fa.nii.gz')
    ret = script_runner.run('scil_compress_streamlines.py', input_fib,
                            'gyri_fanning_c.trk', '-e', '0.1',
                            '--reference', input_fa)
    return ret.success


def test_flip_surface(script_runner):
    # Weird behavior, flip around the origin in RASMM rather than the center of
    # the volume in VOX
    os.chdir(os.path.expanduser(tmp_dir.name))
    input_surf = os.path.join(get_home(), 'surface_vtk_fib',
                              'lhpialt.vtk')
    ret = script_runner.run('scil_flip_surface.py', input_surf, 'rhpialt.vtk',
                            'x')
    return ret.success


def test_flip_volume(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    input_fa = os.path.join(get_home(), 'surface_vtk_fib',
                            'fa.nii.gz')
    ret = script_runner.run('scil_flip_volume.py', input_fa, 'fa_flip.nii.gz',
                            'x')
    return ret.success


def test_flip_streamlines(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    input_fib = os.path.join(get_home(), 'surface_vtk_fib',
                             'gyri_fanning.fib')
    input_fa = os.path.join(get_home(), 'surface_vtk_fib',
                            'fa.nii.gz')
    ret = script_runner.run('scil_flip_streamlines.py', input_fib,
                            'gyri_fanning.tck', 'x', '--reference', input_fa)
    return ret.success


def test_smooth_surface(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    input_surf = os.path.join(get_home(), 'surface_vtk_fib',
                              'lhpialt.vtk')
    ret = script_runner.run('scil_smooth_surface.py', input_surf,
                            'lhpialt_smooth.vtk', '-n', '5', '-s', '1')
    return ret.success


def test_transform_surface(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    input_surf = os.path.join(get_home(), 'surface_vtk_fib',
                              'lhpialt.vtk')
    input_aff = os.path.join(get_home(), 'surface_vtk_fib',
                             'affine.txt')
    ret = script_runner.run('scil_transform_surface.py', input_surf, input_aff,
                            'lhpialt_lin.vtk')
    return ret.success
