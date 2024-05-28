#!/home/ayat-jinane-g2/src/scilpy/myenv/bin/python3.10
# -*- coding: utf-8 -*-
import os

import numpy as np

from dicelib import ui
from dicelib.tractogram import compute_lenghts
from dicelib.ui import ColoredArgParser

# parse the input parameters
parser = ColoredArgParser(description=compute_lenghts.__doc__.split('\n')[0])
parser.add_argument("input_tractogram", help="Input tractogram")
parser.add_argument(
    "output_scalar_file",
    help="Output scalar file (.npy or .txt) that will contain "
         "the streamline lengths")
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

# check for errors
output_scalar_file_ext = os.path.splitext(options.output_scalar_file)[1]
if output_scalar_file_ext not in ['.txt', '.npy']:
    ui.ERROR('Invalid extension for the output scalar file')
if os.path.isfile(options.output_scalar_file) and not options.force:
    ui.ERROR('Output scalar file already exists, use -f to overwrite')

try:
    # call the actual function
    lengths = compute_lenghts(
        options.input_tractogram,
        options.verbose,
    )
    # save the lengths to file
    if output_scalar_file_ext == '.txt':
        np.savetxt(options.output_scalar_file, lengths, fmt='%.4f')
    else:
        np.save(options.output_scalar_file, lengths, allow_pickle=False)

except Exception as e:
    if os.path.isfile(options.output_scalar_file):
        os.remove(options.output_scalar_file)
    ui.ERROR(e.__str__() if e.__str__() else 'A generic error has occurred')
