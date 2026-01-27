# -*- coding: utf-8 -*-

import pytest
from scilpy.utils.orientation import (validate_voxel_order,
                                      parse_voxel_order)


def test_validate_voxel_order_valid():
    """Test that valid axis codes pass validation."""
    validate_voxel_order("RAS")
    validate_voxel_order(("R", "A", "S"))
    validate_voxel_order(["R", "A", "S"])


def test_validate_voxel_order_none():
    """Test that None raises a ValueError."""
    with pytest.raises(ValueError,
                       match="Axis codes cannot be None."):
        validate_voxel_order(None)


def test_validate_voxel_order_invalid_code():
    """Test that an invalid code raises a ValueError."""
    with pytest.raises(ValueError,
                       match="Invalid axis code 'X' in target."):
        validate_voxel_order("XAS")


def test_validate_voxel_order_conflicting_codes():
    """Test that conflicting codes raise a ValueError."""
    with pytest.raises(ValueError,
                       match="Conflicting axis codes 'L' and 'R' in target."):
        validate_voxel_order("LRS")


def test_validate_voxel_order_wrong_length():
    """Test that codes with length != 3 raise a ValueError."""
    with pytest.raises(ValueError,
                       match="Target axis codes must be of length 3."):
        validate_voxel_order("RASL")


def test_validate_voxel_order_repeated_codes():
    """Test that repeated codes raise a ValueError."""
    with pytest.raises(ValueError,
                       match="Target axis codes must be unique."):
        validate_voxel_order("RRS")


def test_parse_voxel_order_valid_alpha():
    """Test parsing of valid alphabetical voxel order strings."""
    assert parse_voxel_order("RAS") == ("R", "A", "S")
    assert parse_voxel_order("LPI") == ("L", "P", "I")
    assert parse_voxel_order("ASR") == ("A", "S", "R")


def test_parse_voxel_order_invalid_alpha_length():
    """Test that alphabetical strings of incorrect length raise an error."""
    with pytest.raises(ValueError,
                       match="Voxel order string must have 3 characters."):
        parse_voxel_order("RA")


def test_parse_voxel_order_valid_numeric():
    """Test parsing of valid numeric voxel order strings."""
    assert parse_voxel_order("1,2,3") == ("R", "A", "S")
    assert parse_voxel_order("-1,2,-3") == ("L", "A", "I")
    assert parse_voxel_order("2,3,1") == ("A", "S", "R")


def test_parse_voxel_order_invalid_numeric_repeat():
    """Test that numeric strings with repeated axes raise an error."""
    with pytest.raises(ValueError, match="Axes cannot be repeated."):
        parse_voxel_order("1,1,2")


def test_parse_voxel_order_invalid_format():
    """Test that mixed or invalid format strings raise an error."""
    with pytest.raises(ValueError,
                       match="Invalid voxel order format: 1A2"):
        parse_voxel_order("1,A,2")

    with pytest.raises(ValueError,
                       match="Invalid numeric voxel order. Must use 1, 2, and 3."):
        parse_voxel_order("1,2,3,4")
