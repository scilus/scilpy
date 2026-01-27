# -*- coding: utf-8 -*-

import re


def validate_voxel_order(axcodes):
    """
    Validate a set of axis codes.
    Parameters
    ----------
    axcodes : str or tuple or list
        The axis codes to validate (e.g., "LPS", ("R", "A", "S")).
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
    return axcodes


def parse_voxel_order(order_str):
    """
    Parse the voxel order string into a tuple of axis codes.
    """
    order_str = order_str.replace(',', '').replace(' ', '')

    if order_str.isalpha():
        if len(order_str) != 3:
            raise ValueError("Voxel order string must have 3 characters.")
        return validate_voxel_order(tuple(order_str.upper()))

    numeric_parts = re.findall(r'-?\d', order_str)
    if len(numeric_parts) == 3:
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
        if len(set(order)) != 3:
            # Handle swapped axes from numeric input (e.g., '231')
            axis_vals = [ras_map[abs(int(p))] for p in numeric_parts]
            if len(set(axis_vals)) == 3:
                return validate_voxel_order(tuple(order))
            else:
                raise ValueError("Invalid numeric voxel order. "
                                 "Axes cannot be repeated.")

        return validate_voxel_order(tuple(order))

    # Handling swapped axes like '231' more robustly
    if order_str.replace('-', '').isdigit():
        # This will handle cases like '213', '312', etc.
        unique_digits = set(abs(int(c)) for c in re.findall(r'-?\d',
                                                            order_str))
        if unique_digits != {1, 2, 3}:
            raise ValueError("Invalid numeric voxel order. Must use 1, 2, "
                             "and 3.")

        # Re-evaluating with a more direct mapping for swaps
        parsed_order = []
        initial_ras = ['R', 'A', 'S']
        flip_map = {'R': 'L', 'A': 'P', 'S': 'I'}

        for part in numeric_parts:
            num = int(part)
            idx = abs(num) - 1
            axis = initial_ras[idx]
            if num < 0:
                axis = flip_map.get(axis, axis)  # Flip if negative
            parsed_order.append(axis)

        # Final validation for unique axes
        final_axes = [a[0] for a in parsed_order]
        if len(set(final_axes)) != 3:
            # Create a more descriptive error for conflicting axes
            # like ('L', 'R', 'S')
            pairs = [('L', 'R'), ('A', 'P'), ('S', 'I')]
            for pair in pairs:
                if pair[0] in final_axes and pair[1] in final_axes:
                    raise ValueError(f"Conflicting axes in input: "
                                     f"{pair[0]} and {pair[1]}")

        return validate_voxel_order(tuple(parsed_order))

    raise ValueError(f"Invalid voxel order format: {order_str}")
