import time
import logging
import numpy as np

from scipy.spatial import KDTree
from scilpy.tracking.fibertube_utils import (streamlines_to_segments,
                                             dist_segment_segment)
from dipy.io.stateful_tractogram import StatefulTractogram
from scilpy.tracking.utils import tqdm_if_verbose


class IntersectionFinder:
    """
    Utility class for finding intersections in a given StatefulTractogram with
    a diameter for each streamline.
    """

    FLOAT_EPSILON = 1e-7

    def __init__(self, in_sft: StatefulTractogram, diameters: list,
                 shuffle_segments=True, rng_seed=0, verbose=False):
        """
        Builds a KDTree from all the tractogram's segments
        and stores data required later for filtering.

        Parameters
        ----------
        in_sft : StatefulTractogram
            Stateful Tractogram object containing streamlines to filter.
        diameters : list
            Diameters of each streamline of the tractogram.
        shuffle_segments: bool
            Should pick streamline segments randomly. If set to false, they
            will be picked in order from the first segment of the first
            streamline to the last segment of the last streamline.
        rng_seed : int
            Seed to be used for random number generation.
        verbose : bool
            Should produce verbose output.
        """
        self.diameters = diameters
        self.max_diameter = np.max(diameters)
        self.rng_seed = rng_seed
        self.verbose = verbose
        self.in_sft = in_sft
        self.streamlines = in_sft.streamlines
        self.seg_centers, self.seg_indices, self.max_seg_length = (
            streamlines_to_segments(self.streamlines, verbose=verbose))
        if shuffle_segments:
            logging.debug("Shuffling streamline segments")
            indexes = list(range(len(self.seg_centers)))
            gen = np.random.default_rng(rng_seed)
            gen.shuffle(indexes)

            self.seg_centers = self.seg_centers[indexes]
            self.seg_indices = self.seg_indices[indexes]

        self.tree = KDTree(self.seg_centers)

        self._invalid = []
        self._collisions = []
        self._obstacle = []
        self._excluded = []

        if self.max_seg_length >= 0.3:
            logging.warning("The longest streamline segment is over 0.3mm. " +
                            "Performance may drop significantly. " +
                            "Resampling to ~0.2mm is recommended. "
                            "(See scil_tractogram_resample_nb_points.py)")

    @property
    def invalid(self):
        """Streamlines that hit another streamline and should be
        filtered out."""
        return self._invalid

    @property
    def collisions(self):
        """Collision point of each invalid streamline."""
        return self._collisions

    @property
    def obstacle(self):
        """Streamlines hit by an invalid streamline. They should not
        be filtered and are saved separately merely for visualization."""
        return self._obstacle

    @property
    def excluded(self):
        """Streamlines that don't collide, but should be excluded for
        other reasons."""
        return self._excluded

    def find_intersections(self, min_distance=0):
        """
        Finds intersections within the initialized data of the object

        Produces and stores:
            invalid : ndarray[bool]
                Bit map identifying streamlines that hit another streamline
                and should be filtered out.
            collisions : ndarray[float32]
                Collision point of each collider.
            obstacle : ndarray[bool]
                Streamlines hit by invalid. They should not be filtered and
                are flagged simply for visualization.
            excluded : ndarray[bool]
                Streamlines that don't collide, but should be excluded for
                other reasons. (ex: distance does not respect min_distance)

        Parameters
        ----------
        min_distance: float
            If set, streamlines will be filtered more
            aggressively so that even if they don\'t collide,
            being below [min_distance] apart (external to their
            diameter) will be interpreted as a collision. This
            option is the same as filtering with a large diameter
            but only saving a small diameter in out_tractogram.
            (Value in mm)
        """
        start_time = time.time()
        streamlines = self.streamlines

        invalid = np.full((len(streamlines)), False, dtype=np.bool_)
        collisions = np.zeros((len(streamlines), 3), dtype=np.float32)
        obstacle = np.full((len(streamlines)), False, dtype=np.bool_)
        excluded = np.full((len(streamlines)), False, dtype=np.bool_)

        # si   : Streamline Index | index of streamline within the tractogram.
        # pi   : Point Index      | index of point coordinate within a
        #                           streamline.
        # segi : Segment Index    | index of streamline segment within the
        #                           entire tractogram.
        for segi, center in tqdm_if_verbose(enumerate(self.seg_centers),
                                            self.verbose,
                                            total=len(self.seg_centers)):
            si = self.seg_indices[segi][0]

            # [Pruning 1] If current streamline has already collided or been
            #             excluded, skip.
            if invalid[si] or excluded[si]:
                continue

            neighbors = self.tree.query_ball_point(
                center,
                self.max_seg_length + self.max_diameter + min_distance,
                workers=-1)

            for neighbor_segi in neighbors:
                neighbor_si = self.seg_indices[neighbor_segi][0]

                # [Pruning 2] Skip if neighbor is our streamline
                if neighbor_si == si:
                    continue

                # [Pruning 3] If neighbor has already collided or been
                #             excluded, skip.
                if invalid[neighbor_si] or excluded[neighbor_si]:
                    continue

                p0 = streamlines[si][self.seg_indices[segi][1]]
                p1 = streamlines[si][self.seg_indices[segi][1] + 1]
                q0 = streamlines[neighbor_si][
                    self.seg_indices[neighbor_segi][1]]
                q1 = streamlines[neighbor_si][
                    self.seg_indices[neighbor_segi][1] + 1]

                rp = self.diameters[si] / 2
                rq = self.diameters[neighbor_si] / 2

                distance, _, p_coll, q_coll = dist_segment_segment(p0, p1,
                                                                   q0, q1)
                external_distance = distance - rp - rq

                if external_distance < 0:
                    invalid[si] = True
                    # Estimate of collision point
                    collisions[si] = (p_coll + q_coll) / 2
                    obstacle[neighbor_si] = True
                    break
                if min_distance != 0 and external_distance < min_distance:
                    excluded[si] = True
                    break

        logging.debug("Finished finding intersections in " +
                      str(round(time.time() - start_time, 2)) + " seconds.")

        self._invalid = invalid
        self._collisions = collisions
        self._obstacle = obstacle
        self._excluded = excluded

    def build_tractograms(self, save_colliding):
        """
        Builds and saves the various tractograms obtained from
        find_intersections().

        Parameters
        ----------
        save_colliding: bool
            If set, will return invalid_sft and obstacle_sft in addition to
            out_sft.

        Return
        ------
        out_sft: StatefulTractogram
            Tractogram containing final streamlines void of collision.
        invalid_sft: StatefulTractogram | None
            Tractogram containing the invalid streamlines that have been
            removed.
        obstacle_sft: StatefulTractogram | None
            Tractogram containing the streamlines that the invalid
            streamlines collided with. May or may not have been removed
            afterwards during filtering.
        """
        out_streamlines = []
        out_diameters = []
        out_collisions = []
        out_invalid = []
        out_obstacle = []

        for si, s in tqdm_if_verbose(enumerate(self.streamlines), self.verbose,
                                     total=len(self.streamlines)):
            if self._invalid[si]:
                out_invalid.append(s)
                out_collisions.append(self._collisions[si])
            elif not self._excluded[si]:
                out_streamlines.append(s)
                out_diameters.append(self.diameters[si])
            if self._obstacle[si]:
                out_obstacle.append(s)

        out_sft = StatefulTractogram.from_sft(
            out_streamlines, self.in_sft,
            data_per_streamline={'diameters': out_diameters})
        if save_colliding:
            invalid_sft = StatefulTractogram.from_sft(
                out_invalid,
                self.in_sft,
                data_per_streamline={'collisions': out_collisions})
            obstacle_sft = StatefulTractogram.from_sft(
                out_obstacle,
                self.in_sft)
            return out_sft, invalid_sft, obstacle_sft

        return out_sft, None, None
