#!/home/ayat-jinane-g2/src/scilpy/myenv/bin/python3.10
# -*- coding: utf-8 -*-
import os

import dicelib.ui as ui
from dicelib.tractogram import filter
from dicelib.ui import ColoredArgParser

# parse the input parameters
parser = ColoredArgParser(description=filter.__doc__.split('\n')[0])
parser.add_argument("input_tractogram", help="Input tractogram")
parser.add_argument("output_tractogram", help="Output tractogram")
parser.add_argument(
    "--minlength",
    type=float,
    help="Keep streamlines with length [in mm] >= this value")
parser.add_argument(
    "--maxlength",
    type=float,
    help="Keep streamlines with length [in mm] <= this value")
parser.add_argument(
    "--minweight",
    type=float,
    help="Keep streamlines with weight >= this value")
parser.add_argument(
    "--maxweight",
    type=float,
    help="Keep streamlines with weight <= this value")
parser.add_argument(
    "--weights_in",
    help="Text file with the input streamline weights")
parser.add_argument(
    "--weights_out",
    help="Text file for the output streamline weights")
parser.add_argument(
    "--random",
    "-r",
    type=float,
    default=1.0,
    help="Randomly discard streamlines: 0=discard all, 1=keep all")
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

# check if the input weights file is valid
if options.weights_in:
    if not os.path.isfile(options.weights_in):
        ui.ERROR(f"Input weights file not found: {options.weights_in}")
    if options.weights_out and os.path.isfile(
            options.weights_out) and not options.force:
        ui.ERROR(
            f"Output weights file already exists: {options.weights_out}")

# check if the input weights file has absolute path and if not, add the
# current working directory
if options.weights_in and not os.path.isabs(options.weights_in):
    options.weights_in = os.path.join(os.getcwd(), options.weights_in)

# check if the output weights file has absolute path and if not, add the
# current working directory
if options.weights_out and not os.path.isabs(options.weights_out):
    options.weights_out = os.path.join(os.getcwd(), options.weights_out)


# call actual function
filter(
    options.input_tractogram,
    options.output_tractogram,
    options.minlength,
    options.maxlength,
    options.minweight,
    options.maxweight,
    options.weights_in,
    options.weights_out,
    options.random,
    options.verbose,
    options.force
)
