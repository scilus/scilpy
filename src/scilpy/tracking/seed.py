# -*- coding: utf-8 -*-
import numpy as np
from typing import Literal

from dipy.io.stateful_tractogram import Space, Origin
from scilpy.tracking.fibertube_utils import sample_cylinder


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


class FibertubeSeedGenerator(SeedGenerator):
    """
    Adaptation of the scilpy.tracking.seed.SeedGenerator interface for
    fibertube tracking. Generates a given number of seed within the first
    segment of a given number of fibertubes.
    """
    def __init__(self, centerlines, diameters, nb_seeds_per_fibertube,
                    local_seeding: Literal['center', 'random']):
        """
        Parameters
        ----------
        centerlines: list
            Tractogram containing the fibertube centerlines
        diameters: list
            Diameters of each fibertube
        nb_seeds_per_fibertube: int
            Number of seeds to be generated in each fibertube origin segment.
        local_seeding: 'center' | 'random'
            Seeding method within a fibertube origin segment
        """
        if local_seeding not in ['center', 'random']:
            raise ValueError("The provided local_seeding parameter is not "
                             "one of the expected choices: " + local_seeding)

        self.space = Space.VOXMM
        self.origin = Origin.NIFTI

        self.centerlines = centerlines
        self.diameters = diameters
        self.nb_seeds_per_fibertube = nb_seeds_per_fibertube
        self.local_seeding = local_seeding

    def init_generator(self, rng_seed, numbers_to_skip):
        """
        Initialize a numpy number generator according to user's parameters.
        Returns also the shuffled index of all fibertubes.

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
        self.generator = np.random.RandomState(rng_seed)

        # 1. Initializing seeding maps indices (shuffling in-place)
        indices = np.arange(len(self.centerlines))
        self.generator.shuffle(indices)

        # 2. Generating the seed for the random sampling.
        # Because FibertubeSeedGenerator uses rejection sampling to seed
        # within a cylinder, we can't predict how many generator calls will
        # be done in each thread to avoid duplicates. We instead generate a
        # single, predictable number used as a seed for the rejection
        # sampling.
        while numbers_to_skip > 100000:
            self.generator.random_sample(100000)
            numbers_to_skip -= 100000
        self.generator.random_sample(numbers_to_skip)
        sampling_rng_seed = self.generator.randint(0, 2**32-1)
        self.sampling_generator = np.random.default_rng(sampling_rng_seed)

        return self.sampling_generator, indices

    def get_next_pos(self, random_generator: np.random.Generator,
                     shuffled_indices, which_seed):
        which_fi = which_seed // self.nb_seeds_per_fibertube
        fibertube = self.centerlines[shuffled_indices[which_fi]]

        if self.local_seeding == 'center':
            seed = (fibertube[0] + fibertube[1]) / 2

        if self.local_seeding == 'random':
            radius = self.diameters[shuffled_indices[which_fi]] / 2
            seed = sample_cylinder(fibertube[0], fibertube[1], radius, 1,
                                   random_generator)[0]

        return seed[0], seed[1], seed[2]

    def get_next_n_pos(self, random_generator, shuffled_indices,
                       which_seed_start, n):
        which_fi = which_seed_start // self.nb_seeds_per_fibertube
        fibertube = self.centerlines[shuffled_indices[which_fi]]

        if self.local_seeding == 'center':
            seeds = [(fibertube[0] + fibertube[1]) / 2] * n

        if self.local_seeding == 'random':
            radius = self.diameters[shuffled_indices[which_fi]] / 2
            seeds = sample_cylinder(fibertube[0], fibertube[1], radius, n,
                                    random_generator)

        return seeds


class CustomSeedsDispenser(SeedGenerator):
    """
    Adaptation of the scilpy.tracking.seed.SeedGenerator interface for
    using already generated, custom seeds.
    """
    def __init__(self, custom_seeds, space=Space('vox'),
                 origin=Origin('center')):
        """
        Custom seeds need to be in the same space and origin as the ODFs used
        for tracking.

        Parameters
        ----------
        custom_seeds: list
            Custom seeding coordinates.
        space: Space (optional)
            The Dipy space in which the seeds were saved.
            Default: Space.Vox or 'vox'
        origin: Origin (optional)
            The Dipy origin in which the seeds were saved.
            Default: Origin.NIFTI or 'center'
        """
        self.origin = origin
        self.space = space
        self.seeds = custom_seeds
        self.i = 0

    def init_generator(self, rng_seed, numbers_to_skip):
        """
        Does not do anything. Simulates SeedGenerator's implementation for
        retro-compatibility.

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
            Empty list for interface retro-compatibility
        """
        self.i = numbers_to_skip

        return np.random.default_rng(rng_seed), []

    def get_next_pos(self, random_generator: np.random.Generator,
                     shuffled_indices, which_seed):
        seed = self.seeds[self.i]
        self.i += 1

        return seed[0], seed[1], seed[2]

    def get_next_n_pos(self, random_generator, shuffled_indices,
                       which_seed_start, n):
        seeds = self.seeds[self.i:self.i+n]
        self.i += n

        return seeds
