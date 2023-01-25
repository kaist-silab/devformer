import pytest
import numpy as np

from src.problems.dpp.simulator import decap_sim


EXPECTED_RESULT = 5.815012


@pytest.mark.parametrize(
    argnames=["probe", "solution", "keep_out"], argvalues=[(23, [1, 5, 7], [2, 3, 10])]
)
def test_decap_sim(probe, solution, keep_out):
    assert np.allclose(decap_sim(probe, solution, keep_out), EXPECTED_RESULT)
