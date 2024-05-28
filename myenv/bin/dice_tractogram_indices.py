#!/home/ayat-jinane-g2/src/scilpy/myenv/bin/python3.10
# -*- coding: utf-8 -*-
import os

import dicelib.ui as ui
from dicelib.tractogram import recompute_indices
from dicelib.ui import ColoredArgParser

import numpy as np

# parse the input parameters
parser = ColoredArgParser(description=recompute_indices.__doc__.split('\n')[0])
parser.add_argument("indices", help="Indices to recompute")
parser.add_argument("dictionary_kept", help="Dictionary of kept streamlines")
parser.add_argument("--output", "-o", dest="indices_recomputed",
                    help="Output indices file")
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

# check if path to input and output files are valid
if not os.path.isfile(options.indices):
    ui.ERROR(f"Input indices file not found: {options.indices}")
if not os.path.isfile(options.dictionary_kept):
    ui.ERROR(f"Input dictionary file not found: {options.dictionary_kept}")
if os.path.isfile(options.indices_recomputed) and not options.force:
    ui.ERROR(
        f"Output indices file already exists: {options.indices_recomputed}")

# call actual function
new_indices = recompute_indices(
    options.indices,
    options.dictionary_kept,
    verbose=options.verbose
)

# save new indices
if options.indices_recomputed:
    np.savetxt(options.indices_recomputed, new_indices, fmt='%d')
