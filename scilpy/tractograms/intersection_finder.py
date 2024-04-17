import time
import math
import logging
import numpy as np

from scipy.spatial import KDTree
from scilpy.tracking.fibertube import (segment_tractogram,
                                       dist_segment_segment)
from dipy.io.stateful_tractogram import StatefulTractogram
from scilpy.io.utils import v_enumerate


class IntersectionFinder:
    """
    Utility class for finding intersections in a given StatefulTractogram with
    a diameter for each streamline.
    """

    FLOAT_EPSILON = 0.0000001

    @property
    def colliders(self):
        """Bitmap identifying streamlines that hit another streamline
        and should be filtered out."""
        return self._colliders

    @property
    def collisions(self):
        """Collision point of each collider."""
        return self._collisions

    @property
    def collided(self):
        """Streamlines hit by colliders. They should not be filtered and
        are saved simply for visualization."""
        return self._collided

    @property
    def excluded(self):
        """Streamlines that don't collide, but should be excluded for
        other reasons. (see --tmv_size_threshold)"""
        return self._excluded

    def __init__(self, in_sft: StatefulTractogram, diameters: list,
                 verbose=False):
        """
        Builds a KDTree from all the tractogram's segments
        and stores data required later for filtering.

        Parameters
        ----------
        tractogram : StatefulTractogram
            Tractogram to be filtered.
        diameters : list
            Diameters of each streamline of the tractogram.
        verbose : bool
            Should produce verbose output.
        """
        self.diameters = diameters
        self.verbose = verbose
        self.in_sft = in_sft
        self.streamlines = in_sft.streamlines
        self.seg_centers, self.seg_indices, self.max_seg_length = (
            segment_tractogram(self.streamlines, verbose))
        self.tree = KDTree(self.seg_centers)

        self._colliders = []
        self._collisions = []
        self._collided = []
        self._excluded = []

    def find_intersections(self, tmv_size_threshold: float = None):
        """
        Finds intersections within the initialized data of the object

        Produces and stores:
            colliders : ndarray[bool]
                Bit map identifying streamlines that hit another streamline
                and should be filtered out.
            collisions : ndarray[float32]
                Collision point of each collider.
            collided : ndarray[bool]
                Streamlines hit by colliders. They should not be filtered and
                are flagged simply for visualization.
            excluded : ndarray[bool]
                Streamlines that don't collide, but should be excluded for
                other reasons. (see --tmv_size_threshold)

        Parameters
        ----------
        tmv_size_threshold: float
            If set, will filter more aggressively so that the true_max_voxel
            metric will be at most a given value. (see ft_fibers_metrics.py)
        """
        start_time = time.time()
        streamlines = self.streamlines

        colliders = np.full((len(streamlines)), False, dtype=np.bool_)
        collisions = np.zeros((len(streamlines), 3), dtype=np.float32)
        collided = np.full((len(streamlines)), False, dtype=np.bool_)
        excluded = np.full((len(streamlines)), False, dtype=np.bool_)

        # si   : Streamline Index | index of streamline within the tractogram.
        # pi   : Point Index      | index of point coordinate within a
        #                           streamline.
        # segi : Segment Index    | index of streamline segment within the
        #                           entire tractogram.
        for segi, center in v_enumerate(self.seg_centers,
                                        self.verbose):
            si = self.seg_indices[segi][0]

            # [Pruning 1] If current streamline has already collided or been
            #             excluded, skip.
            if colliders[si] or excluded[si]:
                continue

            neighbors = self.tree.query_ball_point(center,
                                                   self.max_seg_length,
                                                   workers=-1)

            for neighbor_segi in neighbors:
                neighbor_si = self.seg_indices[neighbor_segi][0]

                # [Pruning 2] Skip if neighbor is our streamline
                if neighbor_si == si:
                    continue

                # [Pruning 3] If neighbor has already collided or been
                #             excluded, skip.
                if colliders[neighbor_si] or excluded[neighbor_si]:
                    continue

                p0 = streamlines[si][self.seg_indices[segi][1]]
                p1 = streamlines[si][self.seg_indices[segi][1] + 1]
                q0 = streamlines[neighbor_si][
                    self.seg_indices[neighbor_segi][1]]
                q1 = streamlines[neighbor_si][
                    self.seg_indices[neighbor_segi][1] + 1]

                rp = self.diameters[si] / 2
                rq = self.diameters[neighbor_si] / 2

                distance, _, p_coll, _ = dist_segment_segment(p0, p1, q0, q1)
                collide = distance <= rp + rq + self.FLOAT_EPSILON
                external_distance = distance - rp - rq

                if collide:
                    colliders[si] = True
                    collisions[si] = p_coll
                    collided[neighbor_si] = True
                    break

                if (tmv_size_threshold is not None and
                        external_distance < (tmv_size_threshold /
                                             (math.sqrt(2) / 2))):
                    excluded[si] = True
                    break

        logging.debug("Finished finding intersections in " +
                      str(round(time.time() - start_time, 2)) + " seconds.")

        self._colliders = colliders
        self._collisions = collisions
        self._collided = collided
        self._excluded = excluded

    def build_tractograms(self, args):
        """
        Builds and saves the various tractograms obtained from
        find_intersections().

        Parameters
        ----------
        args: Namespace
            Parsed arguments. Used to get the 'save_colliders', 'save_collided
            and 'bbox_check' args. See scilpy.io.utils to add the arguments to
            your parser.

        Return
        ------
        sfts: list(StatefulTractogram)
            List of tractograms to be saved. The order is:
            [out_sft, colliders_sft, collided_sft], with the last two being
            optional based on given arguments.
        out_diameters: list
            List of the new diameters for out_sft.
        """
        out_streamlines = []
        out_diameters = []
        out_collisions = []
        out_colliders = []
        out_collided = []

        for si, s in v_enumerate(self.streamlines, self.verbose):
            if self._colliders[si]:
                out_colliders.append(s)
                out_collisions.append(self._collisions[si])
            elif not self._excluded[si]:
                out_streamlines.append(s)
                out_diameters.append(self.diameters[si])

                if self._collided[si]:
                    out_collided.append(s)

        out_sft = StatefulTractogram.from_sft(out_streamlines, self.in_sft)

        sfts = [out_sft]
        if args.save_colliders:
            collider_sft = StatefulTractogram.from_sft(
                out_colliders,
                self.in_sft,
                data_per_streamline={'collisions': out_collisions})
            sfts.append(collider_sft)

        if args.save_collided:
            collided_sft = StatefulTractogram.from_sft(out_collided,
                                                       self.in_sft)
            sfts.append(collided_sft)

        return sfts, out_diameters
