#!/home/ayat-jinane-g2/src/scilpy/myenv/bin/python3.10
# -*- coding: utf-8 -*-
import os

from dicelib import ui
from dicelib.clustering import run_clustering
from dicelib.ui import ColoredArgParser

# parse the input parameters
parser = ColoredArgParser(description=run_clustering.__doc__.split('\n')[0])
parser.add_argument("file_name_in", help="Input tractogram")
parser.add_argument(
    "--atlas",
    "-a",
    help="Atlas used to compute streamlines connectivity")
parser.add_argument(
    "--conn_thr",
    "-t",
    default=2,
    type=float,
    metavar="THR",
    help="Threshold [in mm]")
parser.add_argument("--clust_thr", type=float, help="Threshold [in mm]")
parser.add_argument(
    "--n_pts",
    type=int,
    default=10,
    help="Number of points for the resampling of a streamline")
parser.add_argument(
    "--save_assignments",
    "-s",
    help="Save the cluster assignments to file")
parser.add_argument(
    "--output_folder",
    "-out",
    help="Folder where to save the split clusters")
parser.add_argument("--file_name_out", "-o", default=None,
                    help="Output clustered tractogram")
parser.add_argument(
    "--n_threads",
    type=int,
    help="Number of threads to use to perform clustering")
parser.add_argument("--force", "-f", action="store_true",
                    help="Force overwriting of the output")
parser.add_argument(
    "--verbose",
    "-v",
    default=2,
    type=int,
    help=("Verbose level [ 0 = no output, 1 = only errors/warnings, "
          "2 = errors/warnings and progress, 3 = all messages, no progress, "
          "4 = all messages and progress ]")
        )

options = parser.parse_args()


# check the input parameters

# check if path to input and output files are valid
if not os.path.isfile(options.file_name_in):
    ui.ERROR("Input file does not exist: %s" % options.file_name_in)

if options.file_name_out is not None:
    out_ext = os.path.splitext(options.file_name_out)[1]
    if out_ext not in ['.trk', '.tck']:
        ui.ERROR('Invalid extension for the output tractogram')
    elif os.path.isfile(options.file_name_out) and not options.force:
        ui.ERROR('Output tractogram already exists, use -f to overwrite')

# check if path to save assignments exists and create it if not
if options.save_assignments is not None:
    out_assignment_ext = os.path.splitext(options.save_assignments)[1]
    if out_assignment_ext not in ['.txt', '.npy']:
        ui.ERROR('Invalid extension for the output scalar file')
    elif os.path.isfile(options.save_assignments) and not options.force:
        ui.ERROR('Output scalar file already exists, use -f to overwrite')

    if not os.path.exists(os.path.dirname(options.save_assignments)):
        os.makedirs(os.path.dirname(options.save_assignments))

# check if atlas exists
if options.atlas is not None:
    if not os.path.exists(options.atlas):
        ui.ERROR('Atlas does not exist')

# check if output folder exists
if options.output_folder is not None:
    if not os.path.exists(options.output_folder):
        os.makedirs(options.output_folder)

# check if number of threads is valid
if options.n_threads is not None:
    if options.n_threads < 1:
        ui.ERROR('Number of threads must be at least 1')

# check if connectivity threshold is valid
if options.conn_thr is not None:
    if options.conn_thr < 0:
        ui.ERROR('Connectivity threshold must be positive')

# check if clustering threshold is valid
if options.clust_thr is not None:
    if options.clust_thr < 0:
        ui.ERROR('Clustering threshold must be positive')


def main():
    run_clustering(file_name_in=options.file_name_in,
                   output_folder=options.output_folder,
                   file_name_out=options.file_name_out,
                   atlas=options.atlas,
                   conn_thr=options.conn_thr,
                   clust_thr=options.clust_thr,
                   n_pts=options.n_pts,
                   save_assignments=options.save_assignments,
                   n_threads=options.n_threads,
                   force=options.force,
                   verbose=options.verbose)


if __name__ == "__main__":
    main()
