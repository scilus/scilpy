#!/home/ayat-jinane-g2/src/scilpy/myenv/bin/python3.10
# -*- coding: utf-8 -*-
import os

from dicelib import ui
from dicelib.tractogram import spline_smoothing
from dicelib.ui import ColoredArgParser

# parse the input parameters
parser = ColoredArgParser(description=spline_smoothing.__doc__.split('\n')[0])
parser.add_argument("input_tractogram", help="Input tractogram")
parser.add_argument("output_tractogram", help="Output tractogram")
parser.add_argument(
    "--ratio",
    "-r",
    type=float,
    default=0.25,
    help="Ratio of points to be kept/used as control points")
parser.add_argument(
    "--step",
    "-s",
    type=float,
    default=1.0,
    help="Sampling step for the output streamlines [in mm]")
parser.add_argument(
    "--verbose",
    "-v",
    default=2,
    type=int,
    help=("Verbose level [ 0 = no output, 1 = only errors/warnings, "
          "2 = errors/warnings and progress, 3 = all messages, no progress, "
          "4 = all messages and progress ]")
)
parser.add_argument("--force", "-f", action="store_true",
                    help="Force overwriting of the output")
options = parser.parse_args()

# check if path to input and output files are valid
if not os.path.isfile(options.input_tractogram):
    ui.ERROR(f"Input tractogram file not found: {options.input_tractogram}")
if os.path.isfile(options.output_tractogram) and not options.force:
    ui.ERROR(
        f"Output tractogram file already exists: {options.output_tractogram}")
# check if the output tractogram file has the correct extension
output_tractogram_ext = os.path.splitext(options.output_tractogram)[1]
if output_tractogram_ext not in ['.trk', '.tck']:
    ui.ERROR("Invalid extension for the output tractogram file")

# check if output tractogram file has absolute path and if not, add the
# current working directory
if not os.path.isabs(options.output_tractogram):
    options.output_tractogram = os.path.join(
        os.getcwd(), options.output_tractogram)


# check if ratio and step are valid
if options.ratio < 0.0 or options.ratio > 1.0:
    ui.ERROR("Invalid ratio, must be between 0 and 1")
if options.step <= 0.0:
    ui.ERROR("Invalid step, must be greater than 0")


# call actual function
spline_smoothing(
    options.input_tractogram,
    options.output_tractogram,
    options.ratio,
    options.step,
    options.verbose,
    options.force
)
