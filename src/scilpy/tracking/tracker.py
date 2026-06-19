# -*- coding: utf-8 -*-
from contextlib import nullcontext
import itertools
import logging
import multiprocessing
import os
import sys
from tempfile import TemporaryDirectory
import traceback
from typing import Union
from tqdm import tqdm

import numpy as np
from dipy.data import get_sphere
from dipy.io.stateful_tractogram import Space, Origin
from dipy.reconst.shm import sh_to_sf_matrix
from dipy.tracking.streamlinespeed import compress_streamlines

from scilpy.image.volume_space_management import DataVolume
from scilpy.tracking.propagator import AbstractPropagator, ODFPropagator, PropagationStatus
from scilpy.reconst.utils import find_order_from_nb_coeff
from scilpy.tracking.seed import SeedGenerator
from scilpy.gpuparallel.opencl_utils import CLKernel, CLManager, have_opencl
from scilpy.tracking.utils import TrackingDirection

# For the multi-processing:
# Dictionary. Will contain all parameters necessary for a sub-process
# initialization.
multiprocess_init_args = {}


class Tracker(object):
    def __init__(self, propagator: AbstractPropagator, mask: DataVolume,
                 seed_generator: SeedGenerator, nbr_seeds, min_nbr_pts,
                 max_nbr_pts, max_invalid_dirs, compression_th=0.1,
                 nbr_processes=1, save_seeds=False,
                 mmap_mode: Union[str, None] = None, rng_seed=1234,
                 track_forward_only=False, skip=0, verbose=False,
                 min_iter=100, append_last_point=True, rap=None):
        """
        Parameters
        ----------
        propagator : AbstractPropagator
            Tracking object.
            This tracker will use space and origin defined in the
            propagator.
        mask : DataVolume
            Tracking volume(s).
        seed_generator : SeedGenerator
            Seeding volume.
        nbr_seeds: int
            Number of seeds to create via the seed generator.
        min_nbr_pts: int
            Minimum number of points for streamlines.
        max_nbr_pts: int
            Maximum number of points for streamlines.
        max_invalid_dirs: int
            Number of consecutives invalid directions allowed during tracking.
        compression_th : float,
            Maximal distance threshold for compression. If None, no
            compression is applied.
        nbr_processes: int
            Number of sub processes to use.
        save_seeds: bool
            Whether to save the seeds associated to their respective
            streamlines.
        mmap_mode: str
            Memory-mapping mode. One of {None, 'r+', 'c'}. This value is passed
            to np.load() when loading the raw tracking data from a subprocess.
        rng_seed: int
            The random "seed" for the random generator.
        track_forward_only: bool
            If true, only the forward direction is computed.
        skip: int
            Skip the first N seeds created (and thus N rng numbers). Useful if
            you want to create new streamlines to add to a previously created
            tractogram with a fixed rng_seed. Ex: If tractogram_1 was created
            with nbr_seeds=1,000,000, you can create tractogram_2 with
            skip 1,000,000.
        verbose: bool
            Display tracking progression with TQDM progress bar.
        min_iter: int
            Minimum number of tracked streamlines required to update the
            tracking progression bar.
        append_last_point: bool
            Whether to add the last point (once out of the tracking mask) to
            the streamline or not. Note that points obtained after an invalid
            direction (based on the propagator's definition of invalid; ex
            when angle is too sharp of sh_threshold not reached) are never
            added.
        rap: RAP object
            Intantiated RAP object.
        """
        self.propagator = propagator
        self.rap = rap
        self.mask = mask
        self.seed_generator = seed_generator
        self.nbr_seeds = nbr_seeds
        self.min_nbr_pts = min_nbr_pts
        self.max_nbr_pts = max_nbr_pts
        self.max_invalid_dirs = max_invalid_dirs
        self.compression_th = compression_th
        self.save_seeds = save_seeds
        self.mmap_mode = mmap_mode
        self.rng_seed = rng_seed
        self.track_forward_only = track_forward_only
        self.append_last_point = append_last_point
        self.skip = skip

        # List to store RAP entry/exit coordinates as tuples (coord, type)
        # where type is 1 for entry and 2 for exit
        self.rap_entry_exit_coords = []

        self.origin = self.propagator.origin
        self.space = self.propagator.space
        if self.space == Space.RASMM:
            raise NotImplementedError(
                "This version of the Tracker is not ready to work in RASMM "
                "space.")
        if (seed_generator.origin != propagator.origin or
                seed_generator.space != propagator.space):
            raise ValueError("Seed generator and propagator must work with "
                             "the same space and origin!")

        if self.min_nbr_pts <= 0:
            logging.warning("Minimum number of points cannot be 0. Changed to "
                            "1.")
            self.min_nbr_pts = 1

        if self.mmap_mode not in [None, 'r+', 'c']:
            logging.warning("Memory-mapping mode cannot be {}. Changed to "
                            "None.".format(self.mmap_mode))
            self.mmap_mode = None

        self.nbr_processes = self._set_nbr_processes(nbr_processes)

        self.printing_frequency = 1000
        self.verbose = verbose
        self.min_iter = min_iter

    def save_rap_entry_exit_mask(self, output_path, reference_img):
        """
        Save RAP entry/exit coordinates as a nifti mask.
        Entry points have value 1, exit points have value 2.

        Parameters
        ----------
        output_path : str
            Path to save the nifti mask file.
        reference_img : nibabel.Nifti1Image
            Reference image to get affine and shape for the output mask.
        """
        import nibabel as nib

        if not self.rap_entry_exit_coords:
            logging.warning("No RAP entry/exit coordinates to save.")
            return

        # Create empty mask with same shape as reference
        mask_data = np.zeros(reference_img.shape[:3], dtype=np.uint8)

        # Convert coordinates to voxel space and set mask values
        # Each element is a tuple (coord, coord_type) where coord_type is 1 (entry) or 2 (exit)
        for coord, coord_type in self.rap_entry_exit_coords:
            # Coordinates are already in voxel space (VOX, center)
            # Round to nearest integer voxel
            vox_coord = np.round(coord).astype(int)

            # Check bounds
            if (0 <= vox_coord[0] < mask_data.shape[0] and
                0 <= vox_coord[1] < mask_data.shape[1] and
                0 <= vox_coord[2] < mask_data.shape[2]):
                # Use max to handle overlapping entry/exit points
                # If both entry and exit occur at same voxel, exit (2) will prevail
                mask_data[vox_coord[0], vox_coord[1], vox_coord[2]] = max(
                    mask_data[vox_coord[0], vox_coord[1], vox_coord[2]], coord_type)

        # Create nifti image and save
        mask_img = nib.Nifti1Image(mask_data, reference_img.affine,
                                   reference_img.header)
        nib.save(mask_img, output_path)

        entry_count = sum(1 for _, t in self.rap_entry_exit_coords if t == 1)
        exit_count = sum(1 for _, t in self.rap_entry_exit_coords if t == 2)
        logging.info(f"Saved RAP entry/exit mask to {output_path}")
        logging.info(f"Entry coordinates: {entry_count}, Exit coordinates: {exit_count}")
        logging.info(f"Unique voxels with entry (1): {np.sum(mask_data == 1)}, "
                     f"exit (2): {np.sum(mask_data == 2)}")

    def track(self):
        """
        Generate a set of streamline from seed, mask and odf files.

        Return
        ------
        streamlines: list of numpy.array
            List of streamlines, represented as an array of positions.
        seeds: list of numpy.array
            List of seeding positions, one 3-dimensional position per
            streamline.
        """
        if self.nbr_processes < 2:
            chunk_id = 0
            lines, seeds = self._get_streamlines(chunk_id)
        else:
            # Each process will use get_streamlines_at_seeds
            chunk_ids = np.arange(self.nbr_processes)
            with TemporaryDirectory() as tmpdir:
                # Lock for logging
                lock = multiprocessing.Manager().Lock()
                zipped_chunks = zip(chunk_ids, [lock] * self.nbr_processes)

                pool = self._prepare_multiprocessing_pool(tmpdir)

                lines_per_process, seeds_per_process = zip(*pool.map(
                    self._get_streamlines_sub, zipped_chunks))
                pool.close()
                # Make sure all worker processes have exited before leaving
                # context manager.
                pool.join()
                lines = [line for line in itertools.chain(*lines_per_process)]
                seeds = [seed for seed in itertools.chain(*seeds_per_process)]

        return lines, seeds

    def _set_nbr_processes(self, nbr_processes):
        """
        If user did not define the number of processes, define it automatically
        (or set to 1 -- no multiprocessing -- if we can't).
        """
        if nbr_processes <= 0:
            try:
                nbr_processes = multiprocessing.cpu_count()
            except NotImplementedError:
                logging.warning("Cannot determine number of cpus: "
                                "nbr_processes set to 1.")
                nbr_processes = 1

        if nbr_processes > self.nbr_seeds:
            nbr_processes = self.nbr_seeds
            logging.info("Setting number of processes to {} since there were "
                         "less seeds than processes.".format(nbr_processes))
        return nbr_processes

    def _prepare_multiprocessing_pool(self, tmpdir):
        """
        Prepare multiprocessing pool.

        Data must be carefully managed to avoid corruption with
        multiprocessing.

        Params
        ------
        tmpdir: str
            Path where to save temporarily the data. This will allow clearing
            the data from memory. We will fetch it back later.

        Returns
        -------
        pool: The multiprocessing pool.
        """
        # Using pool with a class method will serialize all parameters
        # in the class, which can be heavy, but it is what we would be
        # doing manually with a static class.
        # Be careful however, parameter changes inside the method will
        # not be kept.

        # Saving data. We will reload it in each process.
        data_file_name = os.path.join(tmpdir, 'data.npy')
        np.save(data_file_name, self.propagator.datavolume.data)

        # Clear data from memory
        self.propagator.reset_data(new_data=None)

        pool = multiprocessing.Pool(
            self.nbr_processes,
            initializer=self._send_multiprocess_args_to_global,
            initargs=({
                'data_file_name': data_file_name,
                'mmap_mode': self.mmap_mode
            },))

        return pool

    @staticmethod
    def _send_multiprocess_args_to_global(init_args):
        """
        Sends subprocess' initialisation arguments to global for easier access
        by the multiprocessing pool.
        """
        global multiprocess_init_args
        multiprocess_init_args = init_args
        return

    def _get_streamlines_sub(self, params):
        """
        multiprocessing.pool.map input function. Calls the main tracking
        method (_get_streamlines) with correct initialization arguments
        (taken from the global variable multiprocess_init_args).

        Parameters
        ----------
        params: Tuple[chunk_id, Lock]
            chunk_id: int, this processes's id.
            Lock: the multiprocessing lock.

        Return
        -------
        lines: list
            List of list of 3D positions (streamlines).
        """
        chunk_id, lock = params
        global multiprocess_init_args

        self._reload_data_for_new_process(multiprocess_init_args)
        try:
            streamlines, seeds = self._get_streamlines(chunk_id, lock)
            return streamlines, seeds
        except Exception as e:
            logging.error("Operation _get_streamlines_sub() failed.")
            traceback.print_exception(*sys.exc_info(), file=sys.stderr)
            raise e

    def _reload_data_for_new_process(self, init_args):
        """
        Once process is started, load back data.

        Params
        ------
        init_args: Iterable
            Args necessary to reset data. In current implementation: a tuple;
            (file where the data is saved, mmap_mode).
        """
        self.propagator.reset_data(np.load(
            init_args['data_file_name'], mmap_mode=init_args['mmap_mode']))

    def _get_streamlines(self, chunk_id, lock=None):
        """
        Tracks the n streamlines associates with current process (identified by
        chunk_id). The number n is the total number of seeds / the number of
        processes. If asked by user, may compress the streamlines and save the
        seeds.

        Parameters
        ----------
        chunk_id: int
            This process ID.
        lock: Lock
            The multiprocessing lock for verbose printing (optional with
            single processing).

        Returns
        -------
        streamlines: list
            The successful streamlines.
        seeds: list
            The list of seeds for each streamline, if self.save_seeds. Else, an
            empty list.
        """
        streamlines = []
        seeds = []

        # Initialize the random number generator to cover multiprocessing,
        # skip, which voxel to seed and the subvoxel random position
        chunk_size = int(self.nbr_seeds / self.nbr_processes)
        first_seed_of_chunk = chunk_id * chunk_size + self.skip
        random_generator, indices = self.seed_generator.init_generator(
            self.rng_seed, first_seed_of_chunk)
        if chunk_id == self.nbr_processes - 1:
            chunk_size += self.nbr_seeds % self.nbr_processes

        # Getting streamlines
        tqdm_text = "#" + "{}".format(chunk_id).zfill(3)

        if self.verbose:
            if lock is None:
                lock = nullcontext()
            with lock:
                p = tqdm(total=chunk_size, desc=tqdm_text, position=chunk_id+1,
                         leave=False)

        for s in range(chunk_size):
            seed = self.seed_generator.get_next_pos(
                random_generator, indices, first_seed_of_chunk + s)

            # Setting the random value.
            # Previous usage (and usage in Dipy) is to set the random seed
            # based on the (real) seed position. However, in the case where we
            # like to have exactly the same seed more than once, this will lead
            # to exactly the same line, even in probabilistic tracking.
            # Changing to seed position + seed number.
            # Then in the case of multiprocessing, adding also a fraction based
            # on current process ID.
            eps = s + chunk_id / (self.nbr_processes + 1)
            line_generator = np.random.default_rng(
                np.abs(hash((seed + (eps, eps, eps), self.rng_seed))))

            # Forward and backward tracking
            line = self._get_line_both_directions(seed, line_generator)

            if line is not None:
                streamline = np.array(line, dtype='float32')

                if self.compression_th is not None:
                    # Compressing. Threshold is in mm. Verifying space.
                    if self.space == Space.VOX:
                        # Equivalent of sft.to_voxmm:
                        streamline *= self.seed_generator.voxres
                        compress_streamlines(streamline, self.compression_th)
                        # Equivalent of sft.to_vox:
                        streamline /= self.seed_generator.voxres
                    else:
                        compress_streamlines(streamline, self.compression_th)

                streamlines.append(streamline)

                if self.save_seeds:
                    seeds.append(np.asarray(seed, dtype='float32'))

            # Note. Option min_iter does not work with manual pbar update.
            # Will verify manually, lower.
            # Fixed choice of value rather than a percentage of the chunk
            # size because our tracker is quite slow.
            if self.verbose and (s + 1) % self.min_iter == 0:
                with lock:
                    p.update(self.min_iter)

        if self.verbose:
            with lock:
                p.close()
        return streamlines, seeds

    def _get_line_both_directions(self, seeding_pos, line_generator):
        """
        Generate a streamline from an initial position following the tracking
        parameters.

        Parameters
        ----------
        seeding_pos : tuple
            3D position, the seed position.

        Returns
        -------
        line: list of 3D positions
            The generated streamline for seeding_pos.
        """
        # Forward
        line = [np.asarray(seeding_pos)]
        tracking_info = self.propagator.prepare_forward(seeding_pos,
                                                        line_generator)
        if tracking_info == PropagationStatus.ERROR:
            # No good tracking direction can be found at seeding position.
            return None
        line = self._propagate_line(line, tracking_info)

        # Backward
        if not self.track_forward_only:
            if len(line) > 1:
                line.reverse()

            tracking_info = self.propagator.prepare_backward(line,
                                                             tracking_info)
            line = self._propagate_line(line, tracking_info)

        # Clean streamline
        if self.min_nbr_pts <= len(line) <= self.max_nbr_pts:
            return line
        return None

    def _propagate_line(self, line, previous_dir):
        """
        Generate a streamline in forward or backward direction from an initial
        position following the tracking parameters.

        Propagation will stop if the current position is out of bounds (mask's
        bounds and data's bounds should be the same) or if mask's value at
        current position is 0 (usual use is with a binary mask but this is not
        mandatory).

        Parameters
        ----------
        line: List[np.ndarrays]
            Beginning of the line to propagate: list of 3D coordinates
            formatted as arrays.
        previous_dir: Any
            Information necessary to know how to propagate. Type: as understood
            by the propagator. Example, with the typical fODF propagator: the
            previous direction of the streamline, v_in, used to define a cone
            theta, of type TrackingDirection.

        Returns
        -------
        line: list of 3D positions
            At minimum, stays as initial line. Or extended with new tracked
            points.
        """
        invalid_direction_count = 0
        propagation_can_continue = True
        in_rap_region = False  # Track whether we're currently in RAP region
        step_count = 0

        while len(line) < self.max_nbr_pts and propagation_can_continue:

            # Call the RAP function if needed. Can advance of as many points
            # as they want.
            is_currently_in_rap = (propagation_can_continue and self.rap and
                                   self.rap.is_in_rap_region(
                                       line[-1], space=self.space, origin=self.origin))

            # Detect entering RAP region
            if is_currently_in_rap and not in_rap_region:
                self.rap_entry_exit_coords.append((line[-1].copy(), 1))  # 1 for entry
                in_rap_region = True
                logging.debug(f"TRACKER ENTERING pos={np.round(line[-1], 2)}")

            if is_currently_in_rap:
                prev_len = len(line)
                line, new_dir, is_line_valid = (
                    self.rap.rap_multistep_propagate(line, previous_dir))
                if not is_line_valid:
                    logging.debug("TRACKER invalid, stop")
                    break
                if len(line) == prev_len:
                    logging.debug("TRACKER no progress, stop")
                    propagation_can_continue = False
                    break
                new_pos = line[-1]

                # Verify that our RAP propagated point stays within the tracking mask
                propagation_can_continue = self._verify_stopping_criteria(line)
                if not propagation_can_continue:
                    logging.debug("TRACKER out of mask, stop.")
                    line.pop()
                    break

                step_count += 1
            else:
                new_pos, new_dir, is_direction_valid = \
                    self.propagator.propagate(line, previous_dir)

                # Verifying if direction is valid
                # If invalid: break. Else, verify tracking mask.
                if is_direction_valid:
                    invalid_direction_count = 0
                else:
                    invalid_direction_count += 1
                    if invalid_direction_count > self.max_invalid_dirs:
                        break
                
                propagation_can_continue = self._verify_stopping_criteria(line + [new_pos])
                if propagation_can_continue or self.append_last_point:
                    line.append(new_pos)

            previous_dir = new_dir

        logging.debug(f"TRACKER end of propagation: {len(line)} total points, last pos={np.round(line[-1], 2)}")
        return line

    def _verify_stopping_criteria(self, line):
        last_pos = line[-1]

        # Checking if out of bound
        if not self.mask.is_coordinate_in_bound(
                *last_pos, space=self.space, origin=self.origin):
            return False

        # Checking if out of mask
        if self.mask.get_value_at_coordinate(
                *last_pos, space=self.space, origin=self.origin) <= 0:
            return False

        return True


class TrackerAdaViT(Tracker):
    """
    SuperTracker is like a regular tracker, but instead of using a single tracking mask,
    it uses a 4D volume containing many tracking masks and tracks only in the union of all
    masks intersecting the streamline trajectory.
    """
    def __init__(self, propagator: AbstractPropagator,
                 tracking_masks: np.ndarray,
                 seed_generator: SeedGenerator, nbr_seeds, min_nbr_pts,
                 max_nbr_pts, max_invalid_dirs, compression_th=0.1,
                 mask_exclude: Union[None, DataVolume] = None,
                 backtrack_n_pts: int=40, nbr_processes=1, save_seeds=False,
                 mmap_mode: Union[str, None] = None, rng_seed=1234,
                 track_forward_only=False, skip=0, verbose=False,
                 min_iter=100, append_last_point=True):
        super().__init__(propagator, None, seed_generator, nbr_seeds,
                         min_nbr_pts, max_nbr_pts, max_invalid_dirs, compression_th,
                         nbr_processes, save_seeds, mmap_mode, rng_seed,
                         track_forward_only, skip, verbose, min_iter,
                         append_last_point, None)
        # tracking masks
        self.tracking_masks = tracking_masks

        self.n_pts_backtrack = backtrack_n_pts
        self.mask_exclude = mask_exclude
        # TODO: Make into a parameter
        self.max_retries = 10

        # assert space
        if self.space != Space.VOX and self.origin != Origin.CENTER:
            raise NotImplementedError("This version of the Tracker only works in VOX space with CENTER origin.")

        if (seed_generator.origin != propagator.origin or seed_generator.space != propagator.space):
            raise ValueError("Seed generator and propagator must work with the same space and origin!")

    def _get_line_both_directions(self, seeding_pos, line_generator):
        """
        Generate a streamline from an initial position following the tracking
        parameters.

        Parameters
        ----------
        seeding_pos : tuple
            3D position, the seed position.

        Returns
        -------
        line: list of 3D positions
            The generated streamline for seeding_pos.
        """
        # Forward
        line = [np.asarray(seeding_pos)]
        seed_tracking_info = self.propagator.prepare_forward(seeding_pos, line_generator)
        if seed_tracking_info == PropagationStatus.ERROR:
            # No good tracking direction can be found at seeding position.
            return None

        tracking_info = seed_tracking_info

        # variables for backtracking (as in mrtrix)
        include = False
        n_pts_backtrack = self.n_pts_backtrack
        retries = 0
        while not include and retries < self.max_retries:
            line = self._propagate_line(line, tracking_info)
            include = self._verify_inclusion_criteria(line)
            if not include:
                retries += 1
                if n_pts_backtrack >= len(line):
                    return None  # not valid and can't backtrack anymore
                line = line[:-n_pts_backtrack]
                if len(line) >= 2:
                    last_dir = line[-1] - line[-2]
                    sphere_ind = self.propagator.sphere.find_closest(last_dir)
                    tracking_info = TrackingDirection(self.propagator.sphere.vertices[sphere_ind],
                                                      sphere_ind)

        if retries >= self.max_retries:
            logging.debug(f"TRACKER forward direction: max retries reached ({self.max_retries}), discarding streamline.")

        # Backward
        if not self.track_forward_only and include:
            if len(line) > 1:
                line.reverse()

            tracking_info = self.propagator.prepare_backward(line, seed_tracking_info)

            # variables for backtracking (as in mrtrix)
            include = False
            retries = 0
            n_pts_backtrack = self.n_pts_backtrack
            while not include and retries < self.max_retries:
                line = self._propagate_line(line, tracking_info)
                include = self._verify_inclusion_criteria(line)
                if not include:
                    retries += 1
                    if n_pts_backtrack >= len(line):
                        return None  # not valid and can't backtrack anymore
                    line = line[:-n_pts_backtrack]
                    if len(line) >= 2:
                        last_dir = line[-1] - line[-2]
                        sphere_ind = self.propagator.sphere.find_closest(last_dir)
                        tracking_info = TrackingDirection(self.propagator.sphere.vertices[sphere_ind],
                                                          sphere_ind)

        # Clean streamline
        if include and (self.min_nbr_pts <= len(line) <= self.max_nbr_pts):
            return line

        # streamline is either not included or too short/long, we discard it
        return None

    def _verify_stopping_criteria(self, line):
        # project line coordinates onto a grid to find which masks we are in
        # TODO: This is nearest neighbour interpolation. Maybe support trilinear also?
        line_mask = np.zeros(self.tracking_masks.shape[:-1], dtype=bool)

        # line is in origin center, so we add 0.5 to get to
        # corner and then floor to get voxel coordinates
        coords = np.floor(np.array(line) + 0.5).astype(int)

        line_mask[coords[:, 0], coords[:, 1], coords[:, 2]] = True
        line_masks_intersection = self.tracking_masks[line_mask]

        matching_tracking_masks = np.all(line_masks_intersection, axis=0)
        return np.any(matching_tracking_masks)

    def _verify_inclusion_criteria(self, line):
        endpoint = line[-1]
        if self.mask_exclude is None:
            return True  # keep all streamlines when no exclusion mask is provided

        include = self.mask_exclude.get_value_at_coordinate(
            *endpoint, space=self.space, origin=self.origin) <= 0.5

        return include


class GPUTracker():
    """
    Perform local tracking on a ODF field inside a binary mask. The tracking is
    executed on the GPU using the OpenCL API. Tracking is performed in voxel
    space with origin `corner`.

    Streamlines are interrupted as soon as they reach maximum length and
    returned even if they end inside the tracking mask. The ODF image is
    interpolated using nearest neighbor interpolation. No backward tracking is
    performed.

    Parameters
    ----------
    sh : ndarray
        Spherical harmonics volume. Ex: ODF or fODF.
    mask : ndarray
        Tracking mask. Tracking stops outside the mask.
    seeds : ndarray (n_seeds, 3)
        Seed positions in voxel space with origin `center`.
    step_size : float
        Step size in voxel space.
    max_nbr_pts : int
        Maximum length of a streamline in voxel space.
    theta : float or list of float, optional
        Maximum angle (degrees) between 2 steps. If a list, a theta
        is randomly drawn from the list for each streamline.
    sh_basis : str, optional
        Spherical harmonics basis.
    is_legacy : bool, optional
        Whether or not the SH basis is in its legacy form.
    batch_size : int, optional
        Approximate size of GPU batches.
    forward_only: bool, optional
        If True, only forward tracking is performed.
    rng_seed : int, optional
        Seed for GPU random number generator. Reproducible across GPU runs,
        but not guaranteed to match CPU tracker results for the same seed.
    sphere : int, optional
        Sphere to use for the tracking.
    algo : {'prob', 'det'}, optional
        GPU tracking mode. `prob` samples directions from the SF and `det`
        follows the maximum SF direction.
    """
    def __init__(self, sh, mask, seeds, step_size, max_nbr_pts,
                 theta=20.0, sf_threshold=0.1, sh_interp='trilinear',
                 sh_basis='descoteaux07', is_legacy=True, batch_size=100000,
                 forward_only=False, rng_seed=None, sphere=None,
                 algo='prob'):
        if not have_opencl:
            raise ImportError('pyopencl is not installed. In order to use'
                              'GPU tracker, you need to install it first.')
        self.sh = sh
        if sh_interp not in ['nearest', 'trilinear']:
            raise ValueError('Invalid SH interpolation mode: {}'
                             .format(sh_interp))
        self.sh_interp_nn = sh_interp == 'nearest'
        self.mask = mask

        self.n_seeds = len(seeds)

        self.seed_batches =\
            np.array_split(seeds + 0.5, np.ceil(len(seeds)/batch_size))

        if sphere is None:
            self.sphere = get_sphere(name="repulsion724")
        else:
            self.sphere = sphere

        # tracking step_size and number of points
        self.step_size = step_size
        self.sf_threshold = sf_threshold
        self.max_strl_points = max_nbr_pts

        # convert theta to array
        self.theta = np.atleast_1d(theta)

        self.sh_basis = sh_basis
        self.is_legacy = is_legacy
        self.forward_only = forward_only
        self.algo = algo

        if self.algo not in ['prob', 'det']:
            raise ValueError("Invalid GPU tracking algorithm '{}'. Expected "
                             "'prob' or 'det'.".format(self.algo))
        self.probabilistic = self.algo == 'prob'
        if rng_seed is None:
            self.rng_seed = int(np.random.default_rng().integers(
                0, np.iinfo(np.uint32).max, dtype=np.uint32))
        else:
            self.rng_seed = int(np.uint32(rng_seed))

    def _get_max_amplitudes(self, B_mat):
        fodf_max = np.zeros(self.mask.shape,
                            dtype=np.float32)
        fodf_max[self.mask > 0] = np.max(self.sh[self.mask > 0].dot(B_mat),
                                         axis=-1)

        return fodf_max

    def __iter__(self):
        return self._track()

    def _track(self):
        """
        GPU streamlines generator yielding streamlines with corresponding
        seed positions one by one.
        """
        # Convert theta to cos(theta)
        max_cos_theta = np.cos(np.deg2rad(self.theta))

        cl_kernel = CLKernel('tracker', 'tracking', 'local_tracking.cl')

        # Set tracking parameters
        cl_kernel.set_define('IM_X_DIM', self.sh.shape[0])
        cl_kernel.set_define('IM_Y_DIM', self.sh.shape[1])
        cl_kernel.set_define('IM_Z_DIM', self.sh.shape[2])
        cl_kernel.set_define('IM_N_COEFFS', self.sh.shape[3])
        cl_kernel.set_define('N_DIRS', len(self.sphere.vertices))

        cl_kernel.set_define('N_THETAS', len(self.theta))
        cl_kernel.set_define('STEP_SIZE', '{:.8f}f'.format(self.step_size))
        cl_kernel.set_define('MAX_LENGTH', self.max_strl_points)
        cl_kernel.set_define('FORWARD_ONLY',
                             'true' if self.forward_only else 'false')
        cl_kernel.set_define('PROBABILISTIC',
                             'true' if self.probabilistic else 'false')
        cl_kernel.set_define('RNG_SEED', '{}u'.format(np.uint32(self.rng_seed)))
        cl_kernel.set_define('SF_THRESHOLD',
                             '{:.8f}f'.format(self.sf_threshold))
        cl_kernel.set_define('SH_INTERP_NN',
                             'true' if self.sh_interp_nn else 'false')

        # Create CL program
        cl_manager = CLManager(cl_kernel)

        # Input buffers
        # Constant input buffers
        cl_manager.add_input_buffer('sh', self.sh)
        cl_manager.add_input_buffer('vertices', self.sphere.vertices)

        sh_order = find_order_from_nb_coeff(self.sh)
        B_mat = sh_to_sf_matrix(self.sphere, sh_order_max=sh_order,
                                basis_type=self.sh_basis,
                                return_inv=False, legacy=self.is_legacy)
        cl_manager.add_input_buffer('b_matrix', B_mat)

        fodf_max = self._get_max_amplitudes(B_mat)
        cl_manager.add_input_buffer('max_amplitudes', fodf_max)
        cl_manager.add_input_buffer('mask', self.mask.astype(np.float32))

        cl_manager.add_input_buffer('max_cos_theta', max_cos_theta)

        cl_manager.add_input_buffer('seeds')

        cl_manager.add_output_buffer('out_strl')
        cl_manager.add_output_buffer('out_lengths')

        # Generate streamlines in batches
        for seed_batch in self.seed_batches:
            # Update buffers
            cl_manager.update_input_buffer('seeds', seed_batch)

            # output streamlines buffer
            cl_manager.update_output_buffer('out_strl',
                                            (len(seed_batch),
                                             self.max_strl_points, 3))
            # output streamlines length buffer
            cl_manager.update_output_buffer('out_lengths',
                                            (len(seed_batch), 1))

            # Run the kernel
            tracks, n_points = cl_manager.run((len(seed_batch), 1, 1))
            n_points = n_points.flatten().astype(np.int16)
            for (strl, seed, n_pts) in zip(tracks, seed_batch, n_points):
                strl = strl[:n_pts]

                # output is yielded so that we can use LazyTractogram.
                # seed and strl with origin center (same as DIPY)
                yield strl - 0.5, seed - 0.5


class MouseTracker():
    def __init__(self, propagator: ODFPropagator, wm_mask: DataVolume,
                 seed_generator: SeedGenerator, nbr_seeds, min_nbr_pts,
                 max_nbr_pts, max_invalid_dirs, compression_th=0.1,
                 nbr_processes=1, save_seeds=False,
                 mmap_mode = None,  rng_seed=1234,
                 track_forward_only=False, skip=0, verbose=False,
                 min_iter=100, append_last_point=True, rap=None,
                 backtrack_nb_pts=10, sigma_backtrack_pts=4.0):
        """
        Tracker for tracer experiments. Similar to Tracker but with additional
        input projection map and white matter mask. The tracking is constrained
        by the white matter mask and the projection map is used as a
        probabilistic map to favor tracking in voxels with higher projection
        density.

        Parameters
        ----------
        propagator : AbstractPropagator
            Tracking object.
            This tracker will use space and origin defined in the
            propagator.
        wm_mask : DataVolume
            White matter mask. Tracking stops outside this mask.
        projection_map : DataVolume
            Projection density map. Used as a probabilistic map to favor tracking
            in voxels with higher projection density.
        seed_generator : SeedGenerator
            Seeding volume.
        nbr_seeds: int
            Number of seeds to create via the seed generator.
        min_nbr_pts: int
            Minimum number of points for streamlines.
        max_nbr_pts: int
            Maximum number of points for streamlines.
        max_invalid_dirs: int
            Number of consecutives invalid directions allowed during tracking.
        compression_th : float,
            Maximal distance threshold for compression. If None, no
            compression is applied.
        nbr_processes: int
            Number of sub processes to use.
        save_seeds: bool
            Whether to save the seeds associated to their respective
            streamlines.
        mmap_mode: str
            Memory-mapping mode. One of {None, 'r+', 'c'}. This value is passed
            to np.load() when loading the raw tracking data from a subprocess.
        rng_seed: int
            The random "seed" for the random generator.
        track_forward_only: bool
            If true, only the forward direction is computed.
        skip: int
            Skip the first N seeds created (and thus N rng numbers). Useful if
            you want to create new streamlines to add to a previously created
            tractogram with a fixed rng_seed. Ex: If tractogram_1 was created
            with nbr_seeds=1,000,000, you can create tractogram_2 with
            skip 1,000,000.
        verbose: bool
            Display tracking progression with TQDM progress bar.
        min_iter: int
            Minimum number of tracked streamlines required to update the
            tracking progression bar.
        append_last_point: bool
            Whether to add the last point (once out of the tracking mask) to
            the streamline or not. Note that points obtained after an invalid
            direction (based on the propagator's definition of invalid; ex
            when angle is too sharp of sh_threshold not reached) are never
            added.
        """
        if nbr_processes > 1:
            raise ValueError("Multiprocessing is not yet implemented for MouseTracker.")
        if mmap_mode is not None:
            logging.warning("Memory-mapping mode is not yet implemented for MouseTracker. Ignoring mmap_mode argument.")

        self.propagator = propagator
        self.wm_mask = wm_mask
        self.rap = rap
        self.seed_generator = seed_generator
        self.sigma_pts = sigma_backtrack_pts
        self.backtrack_nb_pts = backtrack_nb_pts

        # tracking parameters
        self.nbr_seeds = nbr_seeds
        self.min_nbr_pts = min_nbr_pts
        self.max_nbr_pts = max_nbr_pts
        self.max_invalid_dirs = max_invalid_dirs
        self.compression_th = compression_th
        self.save_seeds = save_seeds
        self.rng_seed = rng_seed
        self.track_forward_only = track_forward_only
        self.append_last_point = append_last_point
        self.skip = skip

        self.origin = self.propagator.origin
        self.space = self.propagator.space
        if self.space != Space.VOX and self.origin != Origin.CENTER:
            raise NotImplementedError("This version of the Tracker only works in VOX space with CENTER origin.")

        if (seed_generator.origin != propagator.origin or seed_generator.space != propagator.space):
            raise ValueError("Seed generator and propagator must work with the same space and origin!")

        if self.min_nbr_pts <= 0:
            logging.warning("Minimum number of points cannot be 0. Changed to "
                            "1.")
            self.min_nbr_pts = 1

        self.printing_frequency = 1000
        self.verbose = verbose
        self.min_iter = min_iter
        self.backtrack_count = 0

        self.rap_entry_exit_coords = []

        # in voxels, minimum separation between backtracking
        # endpoint to consider them as different lines
        self.min_sep_backtracking = 5.0

    def track(self):
        """
        Generate a set of streamline from seed, mask and odf files.

        Return
        ------
        streamlines: list of numpy.array
            List of streamlines, represented as an array of positions.
        seeds: list of numpy.array
            List of seeding positions, one 3-dimensional position per
            streamline.
        """
        lines, seeds = self._get_streamlines()
        return lines, seeds

    def _get_streamlines(self):
        """
        Tracks all streamlines. If asked by user, may compress the streamlines
        and save the seeds.

        Returns
        -------
        streamlines: list
            The successful streamlines.
        seeds: list
            The list of seeds for each streamline, if self.save_seeds. Else, an
            empty list.
        """
        streamlines = []
        seeds = []

        # Initialize the random number generator to cover multiprocessing,
        # skip, which voxel to seed and the subvoxel random position
        first_seed_of_chunk = self.skip
        random_generator, indices = self.seed_generator.init_generator(
            self.rng_seed, first_seed_of_chunk)

        # Getting streamlines
        tqdm_text = "#" + "{}".format(0).zfill(3)

        if self.verbose:
            p = tqdm(total=self.nbr_seeds, desc=tqdm_text, position=1, leave=False)

        for s in range(self.nbr_seeds):
            seed = self.seed_generator.get_next_pos(
                random_generator, indices, first_seed_of_chunk + s)

            # Setting the random value.
            eps = s
            line_generator = np.random.default_rng(
                np.abs(hash((seed + (eps, eps, eps), self.rng_seed))))

            # Forward and backward tracking
            lines = self._get_lines(seed, line_generator)

            if lines is not None:
                for line in lines:
                    streamline = np.array(line, dtype='float32')

                    if self.compression_th is not None:
                        # Compressing. Threshold is in mm. Verifying space.
                        if self.space == Space.VOX:
                            # Equivalent of sft.to_voxmm:
                            streamline *= self.seed_generator.voxres
                            compress_streamlines(streamline, self.compression_th)
                            # Equivalent of sft.to_vox:
                            streamline /= self.seed_generator.voxres
                        else:
                            compress_streamlines(streamline, self.compression_th)

                    streamlines.append(streamline)

                    if self.save_seeds:
                        seeds.append(np.asarray(seed, dtype='float32'))

            # Note. Option min_iter does not work with manual pbar update.
            # Will verify manually, lower.
            # Fixed choice of value rather than a percentage of the chunk
            # size because our tracker is quite slow.
            if self.verbose and (s + 1) % self.min_iter == 0:
                p.update(self.min_iter)

        if self.verbose:
            p.close()
            logging.info(f"TRACKER finished tracking {len(streamlines)} streamlines with {self.backtrack_count} backtracking events.")

        return streamlines, seeds

    def _get_lines(self, seeding_pos, line_generator):
        """
        Generate a streamline from an initial position following the tracking
        parameters.

        Parameters
        ----------
        seeding_pos : tuple
            3D position, the seed position.

        Returns
        -------
        line: list of 3D positions
            The generated streamline for seeding_pos.
        """
        # Forward
        line = [np.asarray(seeding_pos)]

        tracking_info = self.propagator.prepare_forward(seeding_pos, line_generator)
        original_tracking_info = tracking_info  # Keep the original tracking info for backward tracking preparation
        if tracking_info == PropagationStatus.ERROR:
            # No good tracking direction can be found at seeding position.
            return None

        lines = []

        lines_fwd = []
        last_endpoint = None
        while tracking_info is not None:
            # propagate the line in forward direction (tracking_info is the initial direction)
            line = self._propagate_line(line, tracking_info)
            if last_endpoint is not None:
                if np.linalg.norm(np.asarray(line[-1]) - np.asarray(last_endpoint)) > self.min_sep_backtracking:
                    lines_fwd.append(line.copy())
            else:
                lines_fwd.append(line.copy())
            last_endpoint = line[-1]
            # Verify if we should backtrack
            tracking_info, line = self._verify_backtracking_criteria(line, line_generator)
        logging.debug(f"TRACKER forward: {len(lines_fwd)} lines")

        # Backward
        if not self.track_forward_only:

            # we can take any line since they all start at the same seed
            tracking_info = self.propagator.prepare_backward(line[::-1], original_tracking_info)

            lines_backward = []
            last_endpoint = None
            line = [np.asarray(seeding_pos)]
            while tracking_info is not None:
                # propagate the line in forward direction (tracking_info is the initial direction)
                line = self._propagate_line(line, tracking_info)
                if last_endpoint is not None:
                    if np.linalg.norm(np.asarray(line[-1]) - np.asarray(last_endpoint)) > self.min_sep_backtracking:
                        lines_backward.append(line.copy())
                else:
                    lines_backward.append(line.copy())
                last_endpoint = line[-1]
                # Verify if we should backtrack
                tracking_info, line = self._verify_backtracking_criteria(line, line_generator)

            logging.debug(f"TRACKER backward: {len(lines_backward)} lines")
            # Combine forward and backward lines, removing duplicates at the seed
            for f_line in lines_fwd:
                for b_line in lines_backward:
                    b_line_reversed = b_line[::-1]
                    line = b_line_reversed[:-1] + f_line  # combine backward and forward lines, removing duplicate seed point
                    if self.min_nbr_pts <= len(line) <= self.max_nbr_pts:
                        lines.append(line.copy())  # update the forward line with the combined line
                    else:
                        logging.debug(f"TRACKER line invalid, stop. {self.min_nbr_pts} <= {len(line)} <= {self.max_nbr_pts}")
        else:
            lines = lines_fwd
        logging.debug(f"TRACKER total: {len(lines)} lines")

        return lines if len(lines) > 0 else None

    def _verify_backtracking_criteria(self, line, line_generator):
        prob_retry = np.exp(-0.5 * len(line)**2 / self.sigma_pts**2)
        if prob_retry < np.random.uniform(0, 1):
            logging.debug(f"TRACKER no backtracking, stop. prob_retry={prob_retry:.4f}")
            return None, line

        self.backtrack_count += 1  # Increment backtrack count
        if len(line) - 1 <= self.backtrack_nb_pts:
            line = [line[0]]
            tracking_info = self.propagator.prepare_forward(line[0], line_generator)
        else:
            line = line[:-self.backtrack_nb_pts]
            last_dir = line[-1] - line[-2]  # Backtracking direction
            ind = self.propagator.sphere.find_closest(last_dir)
            tracking_info = TrackingDirection(self.propagator.sphere.vertices[ind], ind)
        return tracking_info, line

    def _propagate_line(self, line, previous_dir):
        """
        Generate a streamline in forward or backward direction from an initial
        position following the tracking parameters.

        Propagation will stop if the current position is out of bounds (mask's
        bounds and data's bounds should be the same) or if mask's value at
        current position is 0 (usual use is with a binary mask but this is not
        mandatory).

        Parameters
        ----------
        line: List[np.ndarrays]
            Beginning of the line to propagate: list of 3D coordinates
            formatted as arrays.
        previous_dir: Any
            Information necessary to know how to propagate. Type: as understood
            by the propagator. Example, with the typical fODF propagator: the
            previous direction of the streamline, v_in, used to define a cone
            theta, of type TrackingDirection.

        Returns
        -------
        line: list of 3D positions
            At minimum, stays as initial line. Or extended with new tracked
            points.
        """
        invalid_direction_count = 0
        propagation_can_continue = True
        in_rap_region = False  # Track whether we're currently in RAP region
        step_count = 0

        while len(line) < self.max_nbr_pts and propagation_can_continue:

            # Call the RAP function if needed. Can advance of as many points
            # as they want.
            is_currently_in_rap = (propagation_can_continue and self.rap and
                                   self.rap.is_in_rap_region(
                                       line[-1], space=self.space, origin=self.origin))

            # Detect entering RAP region
            if is_currently_in_rap and not in_rap_region:
                self.rap_entry_exit_coords.append((line[-1].copy(), 1))  # 1 for entry
                in_rap_region = True
                logging.debug(f"TRACKER ENTERING pos={np.round(line[-1], 2)}")

            if is_currently_in_rap:
                prev_len = len(line)
                line, new_dir, is_line_valid = (
                    self.rap.rap_multistep_propagate(line, previous_dir))
                if not is_line_valid:
                    logging.debug("TRACKER invalid, stop")
                    break
                if len(line) == prev_len:
                    logging.debug("TRACKER no progress, stop")
                    propagation_can_continue = False
                    break
                new_pos = line[-1]

                # Verify that our RAP propagated point stays within the tracking mask
                propagation_can_continue = self._verify_stopping_criteria(new_pos)
                if not propagation_can_continue:
                    logging.debug("TRACKER out of mask, stop.")
                    line.pop()
                    break

                step_count += 1
            else:
                new_pos, new_dir, is_direction_valid = \
                    self.propagator.propagate(line, previous_dir)

                # Verifying if direction is valid
                # If invalid: break. Else, verify tracking mask.
                if is_direction_valid:
                    invalid_direction_count = 0
                else:
                    invalid_direction_count += 1
                    if invalid_direction_count > self.max_invalid_dirs:
                        break

                propagation_can_continue = self._verify_stopping_criteria(new_pos)
                if propagation_can_continue or self.append_last_point:
                    line.append(new_pos)

            previous_dir = new_dir

        return line

    def _verify_stopping_criteria(self, last_pos):
        # Checking if out of bound
        if not self.wm_mask.is_coordinate_in_bound(
                *last_pos, space=self.space, origin=self.origin):
            return False

        # Checking if out of mask
        if self.wm_mask.get_value_at_coordinate(
                *last_pos, space=self.space, origin=self.origin) <= 0:
            return False

        # If we are still here, we can continue the propagation.
        return True
