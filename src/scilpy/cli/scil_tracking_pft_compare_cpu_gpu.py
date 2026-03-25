#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run CPU and GPU PFT tracking with identical parameters and report agreement.

This utility executes scil_tracking_pft twice:
- once on CPU
- once with --use_gpu

It then compares tractograms and writes summary metrics to a JSON file.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

from scilpy.cli import scil_tracking_pft
from scilpy.io.utils import (add_overwrite_arg, add_verbose_arg,
                             assert_inputs_exist, assert_outputs_exist,
                             assert_headers_compatible)
from scilpy.version import version_string


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=version_string)

    p.add_argument('in_sh', help='Spherical harmonic file (.nii.gz).')
    p.add_argument('in_seed', help='Seeding mask (.nii.gz).')
    p.add_argument('in_map_include', help='PFT include map (.nii.gz).')
    p.add_argument('map_exclude_file', help='PFT exclude map (.nii.gz).')
    p.add_argument('out_report_json', help='Output comparison report (JSON).')

    p.add_argument('--algo', default='prob', choices=['det', 'prob'],
                   help='Tracking algorithm. [%(default)s]')
    p.add_argument('--step', dest='step_size', type=float, default=0.2,
                   help='Step size in mm. [%(default)s]')
    p.add_argument('--min_length', type=float, default=10.0,
                   help='Minimum length in mm. [%(default)s]')
    p.add_argument('--max_length', type=float, default=300.0,
                   help='Maximum length in mm. [%(default)s]')
    p.add_argument('--theta', type=float,
                   help='Maximum angle between steps.')
    p.add_argument('--act', action='store_true',
                   help='Enable ACT stopping criterion.')
    p.add_argument('--sfthres', dest='sf_threshold', type=float, default=0.1,
                   help='Spherical function threshold. [%(default)s]')
    p.add_argument('--sfthres_init', dest='sf_threshold_init',
                   type=float, default=0.5,
                   help='Initial spherical function threshold. [%(default)s]')

    seed_group = p.add_argument_group('Seeding options')
    seed_sub_exclusive = seed_group.add_mutually_exclusive_group()
    seed_sub_exclusive.add_argument('--npv', type=int,
                                    help='Number of seeds per voxel.')
    seed_sub_exclusive.add_argument('--nt', type=int,
                                    help='Total number of seeds.')

    pft_group = p.add_argument_group('PFT options')
    pft_group.add_argument('--particles', type=int, default=15,
                           help='Number of particles. [%(default)s]')
    pft_group.add_argument('--back', dest='back_tracking', type=float,
                           default=2.0,
                           help='Backtracking distance in mm. [%(default)s]')
    pft_group.add_argument('--forward', dest='forward_tracking', type=float,
                           default=1.0,
                           help='Forward-tracking distance in mm. [%(default)s]')

    out_group = p.add_argument_group('Output options')
    out_group.add_argument('--all', dest='keep_all', action='store_true',
                           help='Keep excluded streamlines in both runs.')
    out_group.add_argument('--compress', dest='compress_th', type=float,
                           help='Compression threshold in mm.')
    out_group.add_argument('--seed', type=int,
                           help='Random seed used in both runs.')
    out_group.add_argument('--keep_tmp', action='store_true',
                           help='Keep generated CPU/GPU tractograms.')

    gpu_group = p.add_argument_group('GPU options')
    gpu_group.add_argument('--batch_size', type=int,
                           help='GPU batch size.')
    gpu_group.add_argument('--sh_interp', choices=['trilinear', 'nearest'],
                           help='GPU SH interpolation mode.')
    gpu_group.add_argument('--forward_only', action='store_true',
                           help='GPU forward tracking only.')

    add_verbose_arg(p)
    add_overwrite_arg(p)
    return p


def _length_stats(streamlines):
    if len(streamlines) == 0:
        return {
            'count': 0,
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
        }

    lengths = []
    for sl in streamlines:
        if len(sl) < 2:
            lengths.append(0.0)
            continue
        seg = np.diff(sl, axis=0)
        lengths.append(float(np.sum(np.linalg.norm(seg, axis=1))))

    arr = np.asarray(lengths, dtype=np.float64)
    return {
        'count': int(arr.size),
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
    }


def _seed_key(seed, ndigits=4):
    return tuple(np.round(np.asarray(seed, dtype=np.float64), ndigits))


def _seed_histogram(seeds):
    hist = {}
    for s in seeds:
        k = _seed_key(s)
        hist[k] = hist.get(k, 0) + 1
    return hist


def _seed_agreement(cpu_seeds, gpu_seeds):
    cpu_hist = _seed_histogram(cpu_seeds)
    gpu_hist = _seed_histogram(gpu_seeds)

    cpu_keys = set(cpu_hist.keys())
    gpu_keys = set(gpu_hist.keys())

    inter = cpu_keys.intersection(gpu_keys)
    union = cpu_keys.union(gpu_keys)

    if union:
        jaccard = float(len(inter) / len(union))
    else:
        jaccard = 1.0

    abs_count_diff_sum = 0
    for k in union:
        abs_count_diff_sum += abs(cpu_hist.get(k, 0) - gpu_hist.get(k, 0))

    return {
        'unique_seed_count_cpu': int(len(cpu_keys)),
        'unique_seed_count_gpu': int(len(gpu_keys)),
        'unique_seed_intersection': int(len(inter)),
        'unique_seed_union': int(len(union)),
        'unique_seed_jaccard': jaccard,
        'abs_count_diff_sum': int(abs_count_diff_sum),
        'seed_only_in_cpu': int(len(cpu_keys - gpu_keys)),
        'seed_only_in_gpu': int(len(gpu_keys - cpu_keys)),
    }


def _run_tracking(base_cmd, use_gpu, verbose):
    cmd = list(base_cmd)
    if use_gpu:
        cmd.append('--use_gpu')

    if verbose:
        print('Running:', ' '.join(cmd))

    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            'Tracking command failed (use_gpu={}):\nSTDOUT:\n{}\nSTDERR:\n{}'
            .format(use_gpu, completed.stdout, completed.stderr))


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    required = [args.in_sh, args.in_seed, args.in_map_include,
                args.map_exclude_file]
    assert_inputs_exist(parser, required)
    assert_headers_compatible(parser, required)
    assert_outputs_exist(parser, args, args.out_report_json)

    if args.npv is None and args.nt is None:
        args.npv = 1

    with tempfile.TemporaryDirectory(prefix='pft-cpu-gpu-compare-') as tmpdir:
        cpu_trk = os.path.join(tmpdir, 'cpu.trk')
        gpu_trk = os.path.join(tmpdir, 'gpu.trk')

        script_path = Path(scil_tracking_pft.__file__).resolve()
        base_cmd = [
            sys.executable,
            str(script_path),
            args.in_sh,
            args.in_seed,
            args.in_map_include,
            args.map_exclude_file,
            cpu_trk,
            '--algo', args.algo,
            '--step', str(args.step_size),
            '--min_length', str(args.min_length),
            '--max_length', str(args.max_length),
            '--sfthres', str(args.sf_threshold),
            '--sfthres_init', str(args.sf_threshold_init),
            '--particles', str(args.particles),
            '--back', str(args.back_tracking),
            '--forward', str(args.forward_tracking),
            '--save_seeds',
        ]

        if args.keep_all:
            base_cmd.append('--all')
        if args.theta is not None:
            base_cmd.extend(['--theta', str(args.theta)])
        if args.act:
            base_cmd.append('--act')
        if args.npv is not None:
            base_cmd.extend(['--npv', str(args.npv)])
        if args.nt is not None:
            base_cmd.extend(['--nt', str(args.nt)])
        if args.seed is not None:
            base_cmd.extend(['--seed', str(args.seed)])
        if args.compress_th is not None:
            base_cmd.extend(['--compress', str(args.compress_th)])
        if args.verbose:
            base_cmd.extend(['--verbose', args.verbose])

        cpu_cmd = list(base_cmd)
        cpu_cmd[6] = cpu_trk

        gpu_cmd = list(base_cmd)
        gpu_cmd[6] = gpu_trk
        gpu_cmd.append('--use_gpu')
        if args.batch_size is not None:
            gpu_cmd.extend(['--batch_size', str(args.batch_size)])
        if args.sh_interp is not None:
            gpu_cmd.extend(['--sh_interp', args.sh_interp])
        if args.forward_only:
            gpu_cmd.append('--forward_only')

        _run_tracking(cpu_cmd, use_gpu=False, verbose=bool(args.verbose))
        _run_tracking(gpu_cmd, use_gpu=True, verbose=bool(args.verbose))

        cpu_sft = nib.streamlines.load(cpu_trk)
        gpu_sft = nib.streamlines.load(gpu_trk)

        cpu_streamlines = list(cpu_sft.streamlines)
        gpu_streamlines = list(gpu_sft.streamlines)

        cpu_stats = _length_stats(cpu_streamlines)
        gpu_stats = _length_stats(gpu_streamlines)

        n_cpu = cpu_stats['count']
        n_gpu = gpu_stats['count']
        if n_cpu > 0:
            rel_count_delta = float((n_gpu - n_cpu) / n_cpu)
        else:
            rel_count_delta = 0.0 if n_gpu == 0 else 1.0

        cpu_seeds = cpu_sft.tractogram.data_per_streamline.get('seeds', [])
        gpu_seeds = gpu_sft.tractogram.data_per_streamline.get('seeds', [])

        report = {
            'inputs': {
                'in_sh': args.in_sh,
                'in_seed': args.in_seed,
                'in_map_include': args.in_map_include,
                'map_exclude_file': args.map_exclude_file,
            },
            'params': {
                'algo': args.algo,
                'step_size': args.step_size,
                'min_length': args.min_length,
                'max_length': args.max_length,
                'theta': args.theta,
                'act': args.act,
                'sf_threshold': args.sf_threshold,
                'sf_threshold_init': args.sf_threshold_init,
                'particles': args.particles,
                'back_tracking': args.back_tracking,
                'forward_tracking': args.forward_tracking,
                'npv': args.npv,
                'nt': args.nt,
                'seed': args.seed,
                'compress_th': args.compress_th,
                'batch_size': args.batch_size,
                'sh_interp': args.sh_interp,
                'forward_only': args.forward_only,
            },
            'cpu': cpu_stats,
            'gpu': gpu_stats,
            'streamline_count': {
                'cpu': int(n_cpu),
                'gpu': int(n_gpu),
                'abs_delta': int(n_gpu - n_cpu),
                'rel_delta_vs_cpu': rel_count_delta,
            },
            'seed_agreement': _seed_agreement(cpu_seeds, gpu_seeds),
            'artifacts': {
                'cpu_trk': cpu_trk,
                'gpu_trk': gpu_trk,
                'kept_tmp': bool(args.keep_tmp),
            },
        }

        with open(args.out_report_json, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, sort_keys=True)

        if args.keep_tmp:
            keep_dir = Path(args.out_report_json).resolve().parent
            cpu_keep = keep_dir / 'pft_cpu.trk'
            gpu_keep = keep_dir / 'pft_gpu.trk'
            Path(cpu_trk).replace(cpu_keep)
            Path(gpu_trk).replace(gpu_keep)
            report['artifacts']['cpu_trk'] = str(cpu_keep)
            report['artifacts']['gpu_trk'] = str(gpu_keep)
            with open(args.out_report_json, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, sort_keys=True)


if __name__ == '__main__':
    main()
