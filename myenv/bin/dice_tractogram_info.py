#!/home/ayat-jinane-g2/src/scilpy/myenv/bin/python3.10
# -*- coding: utf-8 -*-
import os

import dicelib.ui as ui
from dicelib.tractogram import info
from dicelib.ui import ColoredArgParser

# parse the input parameters
parser = ColoredArgParser(description=info.__doc__.split('\n')[0])
parser.add_argument("tractogram", help="Input tractogram")
parser.add_argument("--lenghts", "-l", action="store_true",
                    help="Show stats on streamline lenghts")
parser.add_argument("--max_field_length", "-m", type=int,
                    help="Maximum length allowed for printing a field value")

options = parser.parse_args()

# check if path to input and output files are valid
if not os.path.isfile(options.tractogram):
    ui.ERROR(f"Input tractogram file not found: {options.tractogram}")


# call actual function
info(
    options.tractogram,
    options.lenghts,
    options.max_field_length,
)
