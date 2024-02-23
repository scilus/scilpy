# -*- coding: utf-8 -*-
import numpy as np

from dipy.io.stateful_tractogram import Space, Origin


class SeedGenerator:
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
                 space=Space('vox'), origin=Origin('center'), n_repeats=1):
        """
        Parameters
        ----------
        data: np.ndarray
            The data, ex, loaded from nibabel img.get_fdata(). It will be used
            to find all voxels with values > 0, but will not be kept in memory.
        voxres: np.ndarray(3,)
            The pixel resolution, ex, using img.header.get_zooms()[:3].
        n_repeats: int
            Number of times a same seed position is returned.
            If used, we supposed that calls to either get_next_pos or
            get_next_n_pos are used sequentially. Not verified.
        """
        self.voxres = voxres
        self.n_repeats = n_repeats
        self.origin = origin
        self.space = space
        if space == Space.RASMM:
            raise NotImplementedError("We do not support rasmm space.")
        elif space not in [Space.VOX, Space.VOXMM]:
            raise ValueError("Space should be a choice of Dipy Space.")
        if origin not in [Origin.NIFTI, Origin.TRACKVIS]:
            raise ValueError("Origin should be a choice of Dipy Origin.")

        # self.seed are all the voxels where a seed could be placed
        # (voxel space, origin=corner, int numbers).
        self.seeds_vox_corner = np.array(np.where(np.squeeze(data) > 0),
                                         dtype=float).transpose()

        if len(self.seeds_vox_corner) == 0:
            raise ValueError("There are no positive voxels in the seeding "
                             "mask!")

        # We use this to remember last offset if n_repeats > 1:
        self.previous_offset = None

    def get_next_pos(self, random_generator, shuffled_indices, which_seed):
        """
        Generate the next seed position (Space=voxmm, origin=corner).
        See self.init()_generator to get the generator and shuffled_indices.

        To be used with self.n_repeats, we suppose that sequential
        get_next_pos calls are used with sequentials values of which_seed.

        Parameters
        ----------
        random_generator: numpy random generator
            Initialized numpy number generator.
        shuffled_indices: np.array
            Indices of current seeding map.
        which_seed: int
            Seed number to be processed.
            (which_seed // self.n_repeats corresponds to the index of the
            chosen seed in the flattened seeding mask).

        Return
        ------
        seed_pos: tuple
            Position of next seed expressed in mm.
        """
        nb_seed_voxels = len(self.seeds_vox_corner)

        # Voxel selection from the seeding mask
        ind = (which_seed // self.n_repeats) % nb_seed_voxels
        x, y, z = self.seeds_vox_corner[shuffled_indices[ind]]

        if which_seed % self.n_repeats == 0:
            # Subvoxel initial positioning. Right now x, y, z are in vox space,
            # origin=corner, so between 0 and 1.
            r_x, r_y, r_z = random_generator.uniform(0, 1, size=3)
            self.previous_offset = (r_x, r_y, r_z)
        else:
            # Supposing that calls to get_next_pos are used correctly:
            # previous_offset should already exist and correspond to the
            # correct offset.
            r_x, r_y, r_z = self.previous_offset

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

    def get_next_n_pos(self, random_generator, shuffled_indices,
                       which_seed_start, n):
        """
        Generate the next n seed positions. Intended for GPU usage.
        Equivalent to:
        >>> for s in range(which_seed_start, which_seed_start + nb_seeds):
        >>>     self.get_next_pos(..., s)

        See description of get_next_pos for more information.

        To be used with self.n_repeats, we suppose that sequential
        get_next_n_pos calls are used with sequential values of
        which_seed_start (with steps of nb_seeds).

        Parameters
        ----------
        random_generator: numpy random generator
            Initialized numpy number generator.
        shuffled_indices: np.array
            Indices of current seeding map.
        which_seed_start: int
            First seed numbers to be processed.
            (which_seed_start // self.n_repeats corresponds to the index of the
            chosen seed in the flattened seeding mask).
        n: int
            Number of seeds to get.

        Return
        ------
        seeds: List[List]
            Positions of next seeds expressed seed_generator's space and
            origin.
        """
        nb_seed_voxels = len(self.seeds_vox_corner)
        which_seeds = np.arange(which_seed_start, which_seed_start + n)

        # Voxel selection from the seeding mask
        # Same seed is re-used n_repeats times
        inds = (which_seeds // self.n_repeats) % nb_seed_voxels

        # Prepare sub-voxel random movement now (faster out of loop)
        r_x = np.zeros((n,))
        r_y = np.zeros((n,))
        r_z = np.zeros((n,))

        # Find where which_seeds % self.n_repeats == 0
        # Note. If where_new_offsets[0] is False, supposing that calls to
        # get_next_n_pos are used correctly: previous_offset should already
        # exist and correspond to the correct offset.
        where_new_offsets = ~(which_seeds % self.n_repeats).astype(bool)
        ind_first = np.where(where_new_offsets)[0][0]
        if not where_new_offsets[0]:
            assert self.previous_offset is not None

            # First continuing previous_offset.
            r_x[0:ind_first] = self.previous_offset[0]
            r_y[0:ind_first] = self.previous_offset[1]
            r_z[0:ind_first] = self.previous_offset[2]

        # Then starting and repeating new offsets.
        nb_new_offsets = np.sum(where_new_offsets)
        new_offsets_x = random_generator.uniform(0, 1, size=nb_new_offsets)
        new_offsets_y = random_generator.uniform(0, 1, size=nb_new_offsets)
        new_offsets_z = random_generator.uniform(0, 1, size=nb_new_offsets)
        nb_r = n - ind_first
        r_x[ind_first:] = np.repeat(new_offsets_x, self.n_repeats)[:nb_r]
        r_y[ind_first:] = np.repeat(new_offsets_y, self.n_repeats)[:nb_r]
        r_z[ind_first:] = np.repeat(new_offsets_z, self.n_repeats)[:nb_r]

        # Save previous offset for next batch
        self.previous_offset = (r_x[-1], r_y[-1], r_z[-1])

        seeds = []
        # Looping. toDo, see if can be done faster.
        for i in range(len(which_seeds)):
            x, y, z = self.seeds_vox_corner[shuffled_indices[inds[i]]]

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

    def init_generator(self, rng_seed, numbers_to_skip):
        """
        Initialize a numpy number generator according to user's parameters.
        Returns also the suffled index of all voxels.

        The values are not stored in this classed, but are returned to the
        user, who should add them as arguments in the methods
        self.get_next_pos()
        self.get_next_n_pos()
        The use of this is that with multiprocessing, each process may have its
        own generator, with less risk of using the wrong one when they are
        managed by the user.

        Parameters
        ----------
        rng_seed : int
            The "seed" for the random generator.
        numbers_to_skip : int
            Number of seeds (i.e. voxels) to skip. Useful if you want to
            continue sampling from the same generator as in a first experiment
            (with a fixed rng_seed).

        Return
        ------
        random_generator : numpy random generator
            Initialized numpy number generator.
        indices : ndarray
            Shuffled indices of current seeding map, shuffled with current
            generator.
        """
        random_generator = np.random.RandomState(rng_seed)

        # 1. Initializing seeding maps indices (shuffling in-place)
        indices = np.arange(len(self.seeds_vox_corner))
        random_generator.shuffle(indices)

        # 2. Initializing the random generator
        # For reproducibility through multi-processing, skipping random numbers
        # (by producing rand numbers without using them) until reaching this
        # process (i.e this chunk)'s set of random numbers. Producing only
        # 100000 at the time to prevent RAM overuse.
        # (Multiplying by 3 for x,y,z)
        random_numbers_to_skip = numbers_to_skip * 3
        # toDo: see if 100000 is ok, and if we can create something not
        #  hard-coded
        while random_numbers_to_skip > 100000:
            random_generator.random_sample(100000)
            random_numbers_to_skip -= 100000
        random_generator.random_sample(random_numbers_to_skip)

        return random_generator, indices
