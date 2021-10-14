# -*- coding: utf-8 -*-
import numpy as np


class SeedGenerator(object):
    """
    Class to get seeding positions
    """
    def __init__(self, img):
        self.pixdim = img.header.get_zooms()[:3]

        data = img.get_fdata(caching='unchanged', dtype=np.float64)
        self.seeds = np.array(np.where(np.squeeze(data) > 0)).transpose()

    def get_next_pos(self, random_generator, indices, which_seed):
        """
        Generate the next seed position.

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
        len_seeds = len(self.seeds)
        if len_seeds == 0:
            return []

        half_voxel_range = [self.pixdim[0] / 2,
                            self.pixdim[1] / 2,
                            self.pixdim[2] / 2]

        # Voxel selection from the seeding mask
        ind = which_seed % len_seeds
        x, y, z = self.seeds[indices[np.asscalar(ind)]]

        # Subvoxel initial positioning
        r_x = random_generator.uniform(-half_voxel_range[0],
                                       half_voxel_range[0])
        r_y = random_generator.uniform(-half_voxel_range[1],
                                       half_voxel_range[1])
        r_z = random_generator.uniform(-half_voxel_range[2],
                                       half_voxel_range[2])

        return x * self.pixdim[0] + r_x, y * self.pixdim[1] \
            + r_y, z * self.pixdim[2] + r_z

    def init_pos(self, random_initial_value, first_seed_of_chunk):
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
        indices : List
            Indices of current seeding map.
        """
        random_generator = np.random.RandomState(random_initial_value)
        indices = np.arange(len(self.seeds))
        random_generator.shuffle(indices)

        # Skip to the first seed of the current process' chunk,
        # multiply by 3 for x,y,z
        # Divide the generation to prevent RAM overuse
        seed_to_go = np.asscalar(first_seed_of_chunk) * 3
        while seed_to_go > 100000:
            random_generator.rand(100000)
            seed_to_go -= 100000
        random_generator.rand(seed_to_go)

        return random_generator, indices
