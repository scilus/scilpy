# -*- coding: utf-8 -*-

import re


def validate_voxel_order(axcodes, dimensions=3):
    """
    Validate a set of axis codes.
    Parameters
    ----------
    axcodes : str or tuple or list
        The axis codes to validate (e.g., "LPS", ("R", "A", "S")).
    dimensions : int
        The number of dimensions of the image.
    Returns
    -------
    tuple
        A tuple of validated axis codes.
    Raises
    ------
    ValueError
        If the axis codes are invalid.
    """
    if axcodes is None:
        raise ValueError("Axis codes cannot be None.")

    axcodes = tuple(axcodes)
    if len(axcodes) != dimensions:
        raise ValueError(f"Target axis codes must be of length {dimensions}.")

    # Check unique are only valid axis codes
    valid_codes = {"L", "R", "A", "P", "S", "I"}
    if dimensions == 4:
        valid_codes.add("T")
    for code in axcodes:
        if code not in valid_codes:
            raise ValueError(f"Invalid axis code '{code}' in target.")

    # Check no repeated axis codes (LL, RR, etc.)
    if len(set(axcodes)) != dimensions:
        raise ValueError("Target axis codes must be unique.")

    # Check L/R, A/P, S/I pairs are not both present
    pairs = [("L", "R"), ("A", "P"), ("S", "I")]
    for pair in pairs:
        if pair[0] in axcodes and pair[1] in axcodes:
            raise ValueError(f"Conflicting axis codes '{pair[0]}' and "
                             f"'{pair[1]}' in target.")
    return axcodes


def parse_voxel_order(order_str, dimensions=3):
    """
    Parse the voxel order string into a tuple of axis codes.
    """
    order_str_cleaned = order_str.replace(',', '').replace(' ', '')

    if dimensions == 4 and order_str_cleaned.isalpha():
        raise ValueError("Alphabetical voxel order is not supported for 4D "
                         "images. Please use numeric format.")

    if order_str_cleaned.isalpha():
        if len(order_str_cleaned) != 3:
            raise ValueError("Voxel order string must have 3 characters.")
        return validate_voxel_order(tuple(order_str_cleaned.upper()))

    if order_str_cleaned.replace('-', '').isdigit():
        numeric_parts = re.findall(r'-?\d', order_str_cleaned)
        if len(numeric_parts) == 4 and dimensions != 4:
            raise ValueError("4D voxel order is only supported for 4D images.")
        if len(numeric_parts) not in [3, 4]:
            raise ValueError("Voxel order string must have 3 or 4 numbers.")

        if dimensions == 4:
            ras_map = {1: 'R', 2: 'A', 3: 'S', 4: 'T'}
            flip_map = {'R': 'L', 'A': 'P', 'S': 'I', 'T': 'T'}
            if len(numeric_parts) == 4:
                if abs(int(numeric_parts[3])) != 4:
                    raise ValueError("The 4th dimension must be 4 or -4.")
        else:
            ras_map = {1: 'R', 2: 'A', 3: 'S'}
            flip_map = {'R': 'L', 'A': 'P', 'S': 'I'}

        order = []
        for part in numeric_parts:
            num = int(part)
            axis = ras_map[abs(num)]
            if num < 0:
                axis = flip_map[axis]
            order.append(axis)

        # Check for duplicate axes
        if len(set(order)) != len(numeric_parts):
            # Handle swapped axes from numeric input (e.g., '231')
            axis_vals = [ras_map[abs(int(p))] for p in numeric_parts]
            if len(set(axis_vals)) == len(numeric_parts):
                return validate_voxel_order(tuple(order), dimensions=len(numeric_parts))
            else:
                raise ValueError("Invalid numeric voxel order. "
                                 "Axes cannot be repeated.")

        return validate_voxel_order(tuple(order), dimensions=len(numeric_parts))
    
    raise ValueError(f"Invalid voxel order format: {order_str}")
