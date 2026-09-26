import math

import pytest

from findspingroup.core.tolerances import DEFAULT_TOL, Tolerances


@pytest.mark.parametrize("field", ["space", "moment", "m_eig", "occupancy", "m_matrix_tol"])
@pytest.mark.parametrize("value", [-0.01, math.nan, math.inf, -math.inf])
def test_physical_and_numerical_tolerances_reject_invalid_values(field, value):
    with pytest.raises(ValueError, match=field):
        Tolerances(**{field: value})


@pytest.mark.parametrize("field", ["space", "m_eig", "m_matrix_tol"])
def test_search_and_matrix_tolerances_must_be_positive(field):
    with pytest.raises(ValueError, match=field):
        Tolerances(**{field: 0.0})


def test_exact_moment_and_occupancy_matching_can_be_requested():
    tol = Tolerances(moment=0.0, occupancy=0.0)
    assert tol.moment == 0.0
    assert tol.occupancy == 0.0
    assert tol.space == DEFAULT_TOL.space
