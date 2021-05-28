import numpy as np


class Mask():

    """
    Abstract class for tracking mask object
    """

    def isPropagationContinues(self, pos):
        """
        abstract method
        """
        pass

    def isStreamlineIncluded(self, pos):
        """
        abstract method
        """
        pass


class BinaryMask(Mask):

    """
    Mask class for binary mask.
    """

    def __init__(self, tracking_dataset):
        self.m = tracking_dataset
        # force memmap to array. needed for multiprocessing
        self.m.data = np.array(self.m.data)
        ndim = self.m.data.ndim
        if not (ndim == 3 or (ndim == 4 and self.m.data.shape[-1] == 1)):
            raise ValueError('mask cannot be more than 3d')

    def isPropagationContinues(self, pos):
        """
        The propagation continues if the position is whitin the mask.

        Parameters
        ----------
        pos : tuple, 3D positions.

        Returns
        -------
        boolean
        """
        return (self.m.getPositionValue(*pos) > 0
                and self.m.isPositionInBound(*pos))

    def isStreamlineIncluded(self, pos):
        """
        If the propagation stoped, this function determines if the streamline
        is included in the tractogram. Always True for BinaryMask.

        Parameters
        ----------
        pos : tuple, 3D positions.

        Returns
        -------
        boolean
        """
        return True
