import numpy as np
import pytest

from dengue_pipeline import features


def test_centro_invalido():
    with pytest.raises(ValueError):
        features.endemic_channel(None, np.array([0]), np.array([0]), np.array([0]), 5, center="otro")
