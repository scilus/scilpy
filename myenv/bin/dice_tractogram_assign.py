#!/home/ayat-jinane-g2/src/scilpy/myenv/bin/python3.10
# -*- coding: utf-8 -*-
import os
# import nibabel as nib
import time
from concurrent.futures import ThreadPoolExecutor as tdp

import numpy as np
from dicelib import ui
from dicelib.connectivity import assign
from dicelib.lazytractogram import LazyTractogram
from dicelib.ui import ColoredArgParser

# parse the input parameters
parser = ColoredArgParser(description=assign.__doc__.split('\n')[0])
parser.add_argument("input_tractogram", help="Input tractogram")
parser.add_argument(
    "atlas",
    help="Atlas used to compute streamlines assignments")
parser.add_argument(
    "--conn_threshold",
    "-t",
    default=2,
    type=float,
    metavar="THR",
    help="Threshold [in mm]")
parser.add_argument(
    "--save_assignments",
    help="Save the cluster assignments to file")
parser.add_argument(
    "--force",
    "-f",
    action="store_true",
    help="Force overwrite")
parser.add_argument(
    "--verbose",
    "-v",
    default=2,
    type=int,
    help="Verbose level [ 0 = no output, 1 = only errors/warnings, 2 = errors/warnings and progress, 3 = all messages, no progress, 4 = all messages and progress ]")
options = parser.parse_args()


def compute_chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


MAX_THREAD = 3

ui.set_verbose(options.verbose)

out_assignment_ext = os.path.splitext(options.save_assignments)[1]
if out_assignment_ext not in ['.txt', '.npy']:
    ui.ERROR('Invalid extension for the output scalar file')
elif os.path.isfile(options.save_assignments) and not options.force:
    ui.ERROR('Output scalar file already exists, use -f to overwrite')

# check if path to save assignments exists and create it if not
if not os.path.exists(os.path.dirname(options.save_assignments)):
    os.makedirs(os.path.dirname(options.save_assignments))

# check if atlas exists
if not os.path.exists(options.atlas):
    ui.ERROR('Atlas does not exist')

num_streamlines = int(
    LazyTractogram(
        options.input_tractogram,
        mode='r').header["count"])
ui.INFO(f"Computing assignments for {num_streamlines} streamlines")

chunk_size = int(num_streamlines / MAX_THREAD)
chunk_groups = [
    e for e in compute_chunks(
        np.arange(num_streamlines),
        chunk_size)]

chunks_asgn = []
t0 = time.time()


pbar_array = np.zeros(MAX_THREAD, dtype=np.int32)

with ui.ProgressBar(multithread_progress=pbar_array, total=num_streamlines,
                    disable=(options.verbose in [0, 1, 3]), hide_on_exit=True) as pbar:
    with tdp(max_workers=MAX_THREAD) as executor:
        future = [executor.submit(assign,
                                  options.input_tractogram,
                                  pbar_array,
                                  i,
                                  start_chunk=int(chunk_groups[i][0]),
                                  end_chunk=int(chunk_groups[i][len(chunk_groups[i]) - 1] + 1),
                                  gm_map_file=options.atlas,
                                  threshold=options.conn_threshold) for i in range(len(chunk_groups))]
        chunks_asgn = [f.result() for f in future]
        chunks_asgn = [c for f in chunks_asgn for c in f]


t1 = time.time()
ui.INFO(f"Time taken to compute assignments: {np.round((t1-t0),2)} seconds")

if out_assignment_ext == '.txt':
    with open(options.save_assignments, "w") as text_file:
        for reg in chunks_asgn:
            print('%d %d' % (int(reg[0]), int(reg[1])), file=text_file)
else:
    np.save(options.save_assignments, chunks_asgn, allow_pickle=False)
