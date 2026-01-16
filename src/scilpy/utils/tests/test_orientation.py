# -*- coding: utf-8 -*-

import pytest
from scilpy.utils.orientation import validate_axcodes


def test_validate_axcodes_valid():
    """Test that valid axis codes pass validation."""
    validate_axcodes('RAS')
    validate_axcodes(('R', 'A', 'S'))
    validate_axcodes(['R', 'A', 'S'])


def test_validate_axcodes_none():
    """Test that None raises a ValueError."""
    with pytest.raises(ValueError, match="Axis codes cannot be None."):
        validate_axcodes(None)


def test_validate_axcodes_invalid_code():
    """Test that an invalid code raises a ValueError."""
    with pytest.raises(ValueError, match="Invalid axis code 'X' in target."):
        validate_axcodes('XAS')


def test_validate_axcodes_conflicting_codes():
    """Test that conflicting codes raise a ValueError."""
    with pytest.raises(ValueError,
                         match="Conflicting axis codes 'L' and 'R' in target."):
        validate_axcodes('LRS')


def test_validate_axcodes_wrong_length():
    """Test that codes with length != 3 raise a ValueError."""
    with pytest.raises(ValueError, match="Target axis codes must be of length 3."):
        validate_axcodes('RA')
    with pytest.raises(ValueError, match="Target axis codes must be of length 3."):
        validate_axcodes('RASL')


def test_validate_axcodes_repeated_codes():
    """Test that repeated codes raise a ValueError."""
    with pytest.raises(ValueError, match="Target axis codes must be unique."):
        validate_axcodes('RRS')