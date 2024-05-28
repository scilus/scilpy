#!/home/ayat-jinane-g2/src/scilpy/myenv/bin/python3.10
# -*- coding: utf-8 -*-
import os

from dicelib import ui
from dicelib.tractogram import join
from dicelib.ui import ColoredArgParser

# parse the input parameters
parser = ColoredArgParser(description=join.__doc__.join('\n')[0])
parser.add_argument(
    "input_tractograms", 
    nargs='*',
    help="Input tractograms")
parser.add_argument(
    "output_tractogram", 
    help="Output tractogram")
parser.add_argument(
    "--input_weights",
    nargs='*',
    default=[],
    help="Text files with the input streamline weights. NOTE: the order must be the same of the input tractograms")
parser.add_argument(
    "--weights_out",
    help="Text file for the output streamline weights")
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

pwd = os.getcwd()

# check if path to output file is valid
if os.path.isfile(options.output_tractogram) and not options.force:
    ui.ERROR(
        f"Output tractogram file already exists: {options.output_tractogram}")
# check if the output tractogram file has the correct extension
output_tractogram_ext = os.path.splitext(options.output_tractogram)[1]
if output_tractogram_ext != '.tck':
    ui.ERROR('Invalid extension for the output tractogram file, must be ".tck"')
# check if output tractogram file has absolute path and if not, add the
# current working directory
if not os.path.isabs(options.output_tractogram):
    options.output_tractogram = os.path.join(pwd, options.output_tractogram)

# check if enough tractograms are given in input
if len(options.input_tractograms) < 2:
    ui.ERROR("Too few tractograms provided in input, only 2 or more are allowed")
# check if path to input files is valid
for f in options.input_tractograms:
    if not os.path.isfile( f ):
        ui.ERROR(f"Input tractogram file not found: {f}")
    if os.path.splitext(f)[1] != '.tck':
        ui.ERROR(f'Invalid extension for the input tractogram {f}, must be ".tck"')


if options.input_weights:
    for i,w in enumerate(options.input_weights):
        # check if the input weights file is valid
        if not os.path.isfile(w):
            ui.ERROR(f"Input weights file not found: {w}")
        # check if the input weights file has absolute path and if not, add the current working directory
        if not os.path.isabs(w):
            options.input_weights[i] = os.path.join(pwd, w)
    if options.weights_out and os.path.isfile(options.weights_out) and not options.force:
        # check if the output weights file is valid
        ui.ERROR(
            f"Output weights file already exists: {options.weights_out}")
    # check if the output weights file has absolute path and if not, add the current working directory
    if options.weights_out and not os.path.isabs(options.weights_out):
        options.weights_out = os.path.join(pwd, options.weights_out)


# call actual function
join( 
    options.input_tractograms,
    options.output_tractogram, 
    options.input_weights,
    options.weights_out,
    options.verbose,
    options.force
)