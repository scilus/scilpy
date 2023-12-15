# -*- coding: utf-8 -*-

import amico
from os.path import join
import tempfile


def get_evaluator(dwi, scheme_filename, mask, para_diff, iso_diff,
                  lambda1, lambda2, intra_vol_fraction, intra_orientation_dist,
                  kernels_dir=None):

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Setup AMICO
        amico.core.setup()
        ae = amico.Evaluation('.', '.')
        ae.load_data(dwi, scheme_filename, mask_filename=mask)
        # Compute the response functions
        ae.set_model("NODDI")

        ae.model.set(para_diff, iso_diff, intra_vol_fraction,
                    intra_orientation_dist, False)

        ae.set_solver(lambda1=lambda1, lambda2=lambda2)

        ae.set_config('ATOMS_path',
                      kernels_dir or join(tmp_dir.name, 'kernels',
                                          ae.model.id))

        ae.generate_kernels(regenerate=not kernels_dir)

    return ae
