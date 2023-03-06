# -*- coding: utf-8 -*-
import logging

import numpy as np

from dipy.io.stateful_tractogram import Space, Origin


class SeedGenerator(object):
    """
    Class to get seeding positions.

    Generated seeds are in voxmm space, origin=corner. Ex: a seed sampled
    exactly at voxel i,j,k = (0,1,2), with resolution 3x3x3mm will have
    coordinates x,y,z = (0, 3, 6).

    Using get_next_pos, seeds are placed randomly within the voxel. In the same
    example as above, seed sampled in voxel i,j,k = (0,1,2) will be somewhere
    in the range x = [0, 3], y = [3, 6], z = [6, 9].
    """
    def __init__(self, data, voxres,
                 space=Space('vox'), origin=Origin('center')):
        """
        Parameters
        ----------
        data: np.array
            The data, ex, loaded from nibabel img.get_fdata(). It will be used
            to find all voxels with values > 0, but will not be kept in memory.
        voxres: np.array(3,)
            The pixel resolution, ex, using img.header.get_zooms()[:3].
        """
        self.voxres = voxres

        self.origin = origin
        self.space = space
        if space == Space.RASMM:
            raise NotImplementedError("We do not support rasmm space.")

        # self.seed are all the voxels where a seed could be placed
        # (voxel space, origin=corner, int numbers).
        self.seeds_vox = np.array(np.where(np.squeeze(data) > 0),
                                  dtype=float).transpose()

        if len(self.seeds_vox) == 0:
            logging.warning("There are no positive voxels in the seeding "
                            "mask!")

    def get_next_pos(self, random_generator, indices, which_seed):
        """
        Generate the next seed position (Space=voxmm, origin=corner).

        Parameters
        ----------
        random_generator : numpy random generator
            Initialized numpy number generator.
        indices : List
            Indices of current seeding map.
        which_seed : int
            Seed number to be processed.

        Return
        ------
        seed_pos: tuple
            Position of next seed expressed in mm.
        """
        len_seeds = len(self.seeds_vox)
        if len_seeds == 0:
            return []

        # Voxel selection from the seeding mask
        ind = which_seed % len_seeds
        x, y, z = self.seeds_vox[indices[ind]]

        # Subvoxel initial positioning. Right now x, y, z are in vox space,
        # origin=corner, so between 0 and 1.
        r_x = random_generator.uniform(0, 1)
        r_y = random_generator.uniform(0, 1)
        r_z = random_generator.uniform(0, 1)

        # Moving inside the voxel
        x += r_x
        y += r_y
        z += r_z

        if self.origin == Origin('center'):
            # Bound [0, 0, 0] is now [-0.5, -0.5, -0.5]
            x -= 0.5
            y -= 0.5
            z -= 0.5

        if self.space == Space.VOX:
            return x, y, z
        elif self.space == Space.VOXMM:
            return x * self.voxres[0], y * self.voxres[1], z * self.voxres[2]
        else:
            raise NotImplementedError("We do not support rasmm space.")

    def get_next_n_pos(self, random_generator, indices, which_seeds):
        """
        Generate the next n seed positions. Intended for GPU usage.

        Parameters
        ----------
        random_generator : numpy random generator
            Initialized numpy number generator.
        indices : numpy array
            Indices of current seeding map.
        which_seeds : numpy array
            Seed numbers (i.e. IDs) to be processed.

        Return
        ------
        seed_pos: List[List]
            Positions of next seeds expressed seed_generator's space and
            origin.
        """

        len_seeds = len(self.seeds_vox)

        if len_seeds == 0:
            return []

        # Voxel selection from the seeding mask
        inds = which_seeds % len_seeds

        # Sub-voxel initial positioning
        # Prepare sub-voxel random movement now (faster out of loop)
        n = len(which_seeds)
        r_x = random_generator.uniform(0, 1, size=n)
        r_y = random_generator.uniform(0, 1, size=n)
        r_z = random_generator.uniform(0, 1, size=n)

        seeds = []
        # Looping. toDo, see if can be done faster.
        for i in range(len(which_seeds)):
            x, y, z = self.seeds_vox[indices[inds[i]]]

            # Moving inside the voxel
            x += r_x[i]
            y += r_y[i]
            z += r_z[i]

            if self.origin == Origin('center'):
                # Bound [0, 0, 0] is now [-0.5, -0.5, -0.5]
                x -= 0.5
                y -= 0.5
                z -= 0.5

            if self.space == Space.VOX:
                seed = [x, y, z]
            elif self.space == Space.VOXMM:
                seed = [x * self.voxres[0],
                        y * self.voxres[1],
                        z * self.voxres[2]]
            else:
                raise NotImplementedError("We do not support rasmm space.")
            seeds.append(seed)

        return seeds

    def init_generator(self, random_initial_value, first_seed_of_chunk):
        """
        Initialize numpy number generator according to user's parameter
        and indexes from the seeding map.

        Parameters
        ----------
        random_initial_value : int
            The "seed" for the random generator.
        first_seed_of_chunk : int
            Number of seeds to skip (skip parameter + multi-processor skip).

        Return
        ------
        random_generator : numpy random generator
            Initialized numpy number generator.
        indices : ndarray
            Indices of current seeding map.
        """
        random_generator = np.random.RandomState(random_initial_value)

        # 1. Initializing seeding maps indices (shuffling in-place)
        indices = np.arange(len(self.seeds_vox))
        random_generator.shuffle(indices)

        # 2. Initializing the random generator
        # For reproducibility through multi-processing, skipping random numbers
        # (by producing rand numbers without using them) until reaching this
        # process (i.e this chunk)'s set of random numbers. Producing only
        # 100000 at the time to prevent RAM overuse.
        # (Multiplying by 3 for x,y,z)
        random_numbers_to_skip = first_seed_of_chunk * 3
        # toDo: see if 100000 is ok, and if we can create something not
        #  hard-coded
        while random_numbers_to_skip > 100000:
            random_generator.random_sample(100000)
            random_numbers_to_skip -= 100000
        random_generator.random_sample(random_numbers_to_skip)

        return random_generator, indices
