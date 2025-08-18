# -*- coding: utf-8 -*-

import os

from dipy.io.surface import load_surface

from scilpy.io.utils import is_argument_set

def load_surface_with_reference(parser, args, filepath, arg_name=None):
    """
    Parameters
    ----------
    parser: Argument Parser
        Used to print errors, if any.
    args: Namespace
        Parsed arguments. Used to get the 'reference' and 'bbox_check' args.
        See scilpy.io.utils to add the arguments to your parser.
    filepath: str
        Path of the tractogram file.
    arg_name: str, optional
        Name of the reference argument. By default the args.reference is used.
        If arg_name is given, then args.arg_name_ref will be used instead.
    """
    if is_argument_set(args, 'bbox_check'):
        bbox_check = args.bbox_check
    else:
        bbox_check = True

    _, ext = os.path.splitext(filepath)
    vtk_ext = ['.vtk', '.vtp', '.fib', '.ply', '.stl', '.xml', '.obj']
    if ext not in vtk_ext:
        if args.reference is None:
            parser.error('The reference image is required for FreeSurfer '
                         'surfaces.')

        if args.source_space or args.source_origin:
            print('The source space and source origin can not be changed for '
                  'FreeSurfer surfaces. Will be ignored.')
        sfs = load_surface(filepath, args.reference,
                           bbox_valid_check=bbox_check)
    elif ext in vtk_ext:
        if (not is_argument_set(args, 'reference')) or args.reference is None:
            parser.error('--reference is required for this file format '
                         '{}.'.format(filepath))
        else:
            # Dipy takes care of Freesurfer surfaces (ignore space and origin if any)
            sfs = load_surface(filepath, args.reference,
                               from_space=args.source_space,
                               from_origin=args.source_origin,
                               bbox_valid_check=bbox_check)

    else:
        parser.error('{} is an unsupported file format'.format(filepath))

    return sfs