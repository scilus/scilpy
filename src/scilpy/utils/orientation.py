# -*- coding: utf-8 -*-


def validate_axcodes(axcodes):
    """
    Validate a set of axis codes.

    Parameters
    ----------
    axcodes : str or tuple or list
        The axis codes to validate (e.g., "LPS", ("R", "A", "S")).

    Raises
    ------
    ValueError
        If the axis codes are invalid.
    """
    if axcodes is None:
        raise ValueError("Axis codes cannot be None.")

    if len(axcodes) != 3:
        raise ValueError("Target axis codes must be of length 3.")

    # Check unique are only valid axis codes
    valid_codes = {"L", "R", "A", "P", "S", "I"}
    for code in axcodes:
        if code not in valid_codes:
            raise ValueError(f"Invalid axis code '{code}' in target.")

    # Check no repeated axis codes (LL, RR, etc.)
    if len(set(axcodes)) != 3:
        raise ValueError("Target axis codes must be unique.")

    # Check L/R, A/P, S/I pairs are not both present
    pairs = [("L", "R"), ("A", "P"), ("S", "I")]
    for pair in pairs:
        if pair[0] in axcodes and pair[1] in axcodes:
            raise ValueError(f"Conflicting axis codes '{pair[0]}' and "
                             f"'{pair[1]}' in target.")
