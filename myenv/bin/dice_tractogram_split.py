#!/home/ayat-jinane-g2/src/scilpy/myenv/bin/python3.10
# -*- coding: utf-8 -*-
import os

from dicelib import ui
from dicelib.tractogram import split
from dicelib.ui import ColoredArgParser

# parse the input parameters
parser = ColoredArgParser(description=split.__doc__.split('\n')[0])
parser.add_argument("tractogram", help="Input tractogram")
parser.add_argument(
    "assignments",
    help="Text file with the streamline assignments")
parser.add_argument(
    "output_folder",
    nargs='?',
    default='bundles',
    help="Output folder for the splitted tractograms")
parser.add_argument(
    "regions",
    nargs='*',
    default=[],
    help="Streamline connecting the provided region(s) will be extracted")
parser.add_argument(
    "--weights_in",
    "-w",
    default=None,
    help="Text file with the input streamline weights")
parser.add_argument(
    "--max_open",
    "-m",
    type=int,
    help="Maximum number of files opened at the same time")
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
if not os.path.isfile(options.tractogram):
    ui.ERROR(f"Input tractogram file not found: {options.tractogram}")
if not os.path.isfile(options.assignments):
    ui.ERROR(f"Input assignments file not found: {options.assignments}")
if not os.path.isdir(options.output_folder):
    os.makedirs(options.output_folder)
if not os.path.isfile(options.assignments):
    ui.ERROR(f"Input assignments file not found: {options.assignments}")
if options.weights_in is not None:
    if not os.path.isfile(options.weights_in):
        ui.ERROR(f"Input weights file not found: {options.weights_in}")
if options.force:
    ui.WARNING("Overwriting existing files")

if len(options.regions) > 2:
    ui.ERROR("Too many regions provided, only 2 are allowed")
if len(options.regions) == 0:
    regions = []
else:
    regions = [int(i) for i in options.regions]

# call actual function
split(
    options.tractogram,
    options.assignments,
    options.output_folder,
    regions,
    options.weights_in,
    options.max_open,
    options.verbose,
    options.force
)
