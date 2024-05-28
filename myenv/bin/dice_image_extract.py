#!/home/ayat-jinane-g2/src/scilpy/myenv/bin/python3.10
# -*- coding: utf-8 -*-

import os

from dicelib import ui
from dicelib.image import extract
from dicelib.ui import ColoredArgParser

# parse the input parameters
parser = ColoredArgParser(description=extract.__doc__.split('\n')[0])
parser.add_argument("input_dwi", help="Input DWI data")
parser.add_argument("input_scheme", help="Input scheme")
parser.add_argument("output_dwi", help="Output DWI data")
parser.add_argument("output_scheme", help="Output scheme")
parser.add_argument(
    "--b",
    "-b",
    type=float,
    nargs='+',
    required=True,
    help="List of b-values to extract")
parser.add_argument(
    "--round",
    "-r",
    type=float,
    default=0.0,
    help="Round b-values to nearest integer multiple of this value")
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
if not os.path.isfile(options.input_dwi):
    ui.ERROR(f"Input DWI data file not found: {options.input_dwi}")
if not os.path.isfile(options.input_scheme):
    ui.ERROR(f"Input scheme file not found: {options.input_scheme}")
if os.path.isfile(options.output_dwi) and not options.force:
    ui.ERROR(f"Output DWI data file already exists: {options.output_dwi} "
             "use -f to overwrite")
if os.path.isfile(options.output_scheme) and not options.force:
    ui.ERROR(f"Output scheme file already exists: {options.output_scheme} "
             "use -f to overwrite")

# check if b-values and round are valid
if options.round < 0.0:
    ui.ERROR(f"Round value must be >= 0.0: {options.round}")
if len(options.b) == 0:
    ui.ERROR("No b-values specified")
for b in options.b:
    if b < 0.0:
        ui.ERROR(f"b-value must be >= 0.0: {b}")


# call actual function
extract(
    options.input_dwi,
    options.input_scheme,
    options.output_dwi,
    options.output_scheme,
    options.b,
    options.round,
    options.verbose,
    options.force
)
