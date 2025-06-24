import numpy as np
import pytest
from config.spots import SpotConfig
from data_generation.spots import SpotGenerator


def test_spot_generator_initialization():
    """Tests that the generator initializes the correct number of spots."""
    spot_cfg = SpotConfig(count=10)
    img_shape = (100, 100)
    generator = SpotGenerator(spot_cfg, img_shape)

    assert generator.n_spots == 10
    assert len(generator.coords) == 10
    assert len(generator.intensities) == 10
    assert all(0 <= y < 100 and 0 <= x < 100 for y, x in generator.coords)


def test_moving_spots_update():
    """Tests that the update method moves the spots."""
    spot_cfg = SpotConfig(count=5, max_step=10)  # max_step makes them moving
    img_shape = (100, 100)
    generator = SpotGenerator(spot_cfg, img_shape)

    initial_coords = list(generator.coords)
    generator.update()
    updated_coords = list(generator.coords)

    assert initial_coords != updated_coords  # They should have moved


def test_fixed_spots_no_update():
    """Tests that update does nothing if max_step is None."""
    spot_cfg = SpotConfig(count=5, max_step=None)  # No max_step means fixed
    img_shape = (100, 100)
    generator = SpotGenerator(spot_cfg, img_shape)

    initial_coords = list(generator.coords)
    generator.update()
    updated_coords = list(generator.coords)

    assert initial_coords == updated_coords  # They should NOT have moved


def test_generator_apply_calls_draw(mocker):
    """
    Tests that the apply method calls the drawing function with the correct state.
    We use a mocker to 'spy' on the drawing function.
    """
    # Mock the external draw_spots function
    mock_draw = mocker.patch("data_generation.spots.SpotGenerator.draw_spots", return_value=np.zeros((10, 10)))

    spot_cfg = SpotConfig(count=5, sigma=1.23)
    img_shape = (10, 10)
    generator = SpotGenerator(spot_cfg, img_shape)

    img_in = np.zeros((10, 10))
    generator.apply(img_in)

    # Assert that our mocked function was called once
    mock_draw.assert_called_once()
    # Assert it was called with the correct arguments from the generator's state
    args, kwargs = mock_draw.call_args
    assert args[0] is img_in
    assert args[1] == generator.coords
    assert args[5] == 1.23  # Check the sigma value