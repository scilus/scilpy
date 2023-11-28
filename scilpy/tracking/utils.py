# -*- coding: utf-8 -*-
from scilpy.io.utils import add_sh_basis_args, add_overwrite_arg


class TrackingDirection(list):
    """
    Tracking direction use as 3D cartesian direction (list(x,y,z))
    and has an index to work with discrete sphere.
    """

    def __init__(self, cartesian, index=None):
        super(TrackingDirection, self).__init__(cartesian)
        self.index = index


def add_mandatory_options_tracking(p):
    p.add_argument('in_odf',
                   help='File containing the orientation diffusion function \n'
                        'as spherical harmonics file (.nii.gz). Ex: ODF or '
                        'fODF.')
    p.add_argument('in_seed',
                   help='Seeding mask (.nii.gz).')
    p.add_argument('in_mask',
                   help='Tracking mask (.nii.gz).\n'
                        'Tracking will stop outside this mask. The last point '
                        'of each \nstreamline (triggering the stopping '
                        'criteria) IS added to the streamline.')
    p.add_argument('out_tractogram',
                   help='Tractogram output file (must be .trk or .tck).')


def add_tracking_options(p):
    track_g = p.add_argument_group('Tracking options')
    track_g.add_argument('--step', dest='step_size', type=float, default=0.5,
                         help='Step size in mm. [%(default)s]')
    track_g.add_argument('--min_length', type=float, default=10.,
                         metavar='m',
                         help='Minimum length of a streamline in mm. '
                              '[%(default)s]')
    track_g.add_argument('--max_length', type=float, default=300.,
                         metavar='M',
                         help='Maximum length of a streamline in mm. '
                              '[%(default)s]')
    track_g.add_argument('--theta', type=float,
                         help='Maximum angle between 2 steps. If the angle is '
                              'too big, streamline is \nstopped and the '
                              'following point is NOT included.\n'
                              '["eudx"=60, "det"=45, "prob"=20, "ptt"=20]')
    track_g.add_argument('--sfthres', dest='sf_threshold', metavar='sf_th',
                         type=float, default=0.1,
                         help='Spherical function relative threshold. '
                              '[%(default)s]')
    add_sh_basis_args(track_g)

    return track_g


def add_tracking_ptt_options(p):
    track_g = p.add_argument_group('PTT options')
    track_g.add_argument('--probe_length', dest='probe_length',
                         type=float, default=1.0,
                         help='The length of the probes. Shorter probe_length '
                              + 'yields more dispersed fibers. [%(default)s]')
    track_g.add_argument('--probe_radius', dest='probe_radius',
                         type=float, default=0,
                         help='The radius of the probe. A large probe_radius '
                              + 'helps mitigate noise in the pmf but it might '
                              + 'make it harder to sample thin and intricate '
                              + 'connections, also the boundary of fiber '
                              + 'bundles might be eroded. [%(default)s]')
    track_g.add_argument('--probe_quality', dest='probe_quality',
                         type=int, default=3,
                         help='The quality of the probe. This parameter sets '
                              + 'the number of segments to split the cylinder '
                              + 'along the length of the probe (minimum=2) '
                              + '[%(default)s]')
    track_g.add_argument('--probe_count', dest='probe_count',
                         type=int, default=1,
                         help='The number of probes. This parameter sets the '
                              + 'number of parallel lines used to model the '
                              + 'cylinder (minimum=1). [%(default)s]')
    track_g.add_argument('--data_support_exponent', dest='support_exponent',
                         type=float, default=3,
                         help='Data support to the power dataSupportExponent '
                              + 'is used for rejection sampling.'
                              + '[%(default)s]')

    return track_g


def add_seeding_options(p):
    seed_group = p.add_argument_group(
        'Seeding options', 'When no option is provided, uses --npv 1.')
    seed_sub_exclusive = seed_group.add_mutually_exclusive_group()
    seed_sub_exclusive.add_argument('--npv', type=int,
                                    help='Number of seeds per voxel.')
    seed_sub_exclusive.add_argument('--nt', type=int,
                                    help='Total number of seeds to use.')


def add_out_options(p):
    out_g = p.add_argument_group('Output options')
    out_g.add_argument('--compress', type=float, metavar='thresh',
                       help='If set, will compress streamlines. The parameter '
                            'value is the \ndistance threshold. A rule of '
                            'thumb is to set it to 0.1mm for \ndeterministic '
                            'streamlines and 0.2mm for probabilitic '
                            'streamlines.')
    add_overwrite_arg(out_g)
    out_g.add_argument('--save_seeds', action='store_true',
                       help='If set, save the seeds used for the tracking \n '
                            'in the data_per_streamline property.\n'
                            'Hint: you can then use '
                            'scilpy_compute_seed_density_map.')
    return out_g


def verify_streamline_length_options(parser, args):
    if not args.min_length >= 0:
        parser.error('min_length must be >= 0, but {}mm was provided.'
                     .format(args.min_length))
    if args.max_length < args.min_length:
        parser.error('max_length must be > than min_length, but '
                     'min_length={}mm and max_length={}mm.'
                     .format(args.min_length, args.max_length))


def verify_seed_options(parser, args):
    if args.npv and args.npv <= 0:
        parser.error('Number of seeds per voxel must be > 0.')

    if args.nt and args.nt <= 0:
        parser.error('Total number of seeds must be > 0.')
