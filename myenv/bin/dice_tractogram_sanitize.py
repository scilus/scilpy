#!/home/ayat-jinane-g2/src/scilpy/myenv/bin/python3.10
# -*- coding: utf-8 -*-
from dicelib.ui import ColoredArgParser
from dicelib.tractogram import sanitize

# parse the input parameters
parser = ColoredArgParser(description=sanitize.__doc__.split('\n')[0])
parser.add_argument(
    "input_tractogram", 
    help="Input tractogram")
parser.add_argument(
    "gray_matter", 
    help="Gray matter")
parser.add_argument(
    "white_matter", 
    help="White matter")
parser.add_argument(
    "--output_tractogram", 
    "-out", 
    help="Output tractogram (if None: '_sanitized' appended to the input filename)")
parser.add_argument(
    "--step", 
    type=float, 
    default=0.2, 
    help="Step size [in mm]")
parser.add_argument(
    "--max_dist", 
    type=float, 
    default=2, 
    help="Maximum distance [in mm]")
parser.add_argument(
    "--save_connecting_tck", 
    "-conn", 
    action="store_true", 
    default=False, 
    help="Save also tractogram with only the actual connecting streamlines (if True: '_only_connecting' appended to the output filename)")
parser.add_argument(
    "--verbose", 
    "-v",
    default=2,
    type=int,
    help=("Verbose level [ 0 = no output, 1 = only errors/warnings, "
          "2 = errors/warnings and progress, 3 = all messages, no progress, "
          "4 = all messages and progress ]")
)
parser.add_argument(
    "--force", 
    "-f", 
    action="store_true", 
    help="Force overwriting of the output")
options = parser.parse_args()

# call actual function
sanitize(
    options.input_tractogram,
    options.gray_matter,
    options.white_matter,
    options.output_tractogram,
    options.step,
    options.max_dist,
    options.save_connecting_tck,
    options.verbose,
    options.force
)