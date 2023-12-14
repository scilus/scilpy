# -*- coding: utf-8 -*-
from dipy.io.stateful_tractogram import Space, Origin
import numpy as np

from scilpy.tracking.seed import SeedGenerator


def test_seed_generation():

    mask = np.zeros((5, 5, 5))

    # Two seeds:
    mask[1, 1, 1] = 1
    mask[4, 3, 2] = 1

    # With n_repeats: it is our role to get seeds sequentially.
    generator = SeedGenerator(mask, voxres=[1, 1, 1], space=Space('vox'),
                              origin=Origin('corner'), n_repeats=3)

    rng_generator, shuffled_indices = generator.init_generator(
        rng_seed=1, numbers_to_skip=0)

    # Generating 3 seeds, should be the same
    seed0 = generator.get_next_pos(rng_generator, shuffled_indices, 0)
    seed1 = generator.get_next_pos(rng_generator, shuffled_indices, 1)
    seed2 = generator.get_next_pos(rng_generator, shuffled_indices, 2)

    assert np.array_equal(seed0, seed1)
    assert np.array_equal(seed0, seed2)
    assert np.array_equal(np.floor(seed0), [1, 1, 1])

    # Generating more seeds, should be the same, in the other voxel
    seed3 = generator.get_next_pos(rng_generator, shuffled_indices, 3)
    seed4 = generator.get_next_pos(rng_generator, shuffled_indices, 4)
    _ = generator.get_next_pos(rng_generator, shuffled_indices, 5)
    assert np.array_equal(seed3, seed4)
    assert np.array_equal(np.floor(seed3), [4, 3, 2])

    # Generating n=4 seeds at once, back to voxel 1
    seeds = generator.get_next_n_pos(rng_generator, shuffled_indices, 6, 4)
    assert np.array_equal(seeds[0], seeds[1])
    assert np.array_equal(seeds[0], seeds[2])
    assert np.array_equal(np.floor(seeds[0]), [1, 1, 1])
    assert np.array_equal(np.floor(seeds[3]), [4, 3, 2])

