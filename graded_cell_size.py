from functools import partial
from typing import Callable

import numpy as np
import pyvista as pv

from microgen import Tpms
from microgen.shape.surface_functions import gyroid


def sigmoid(x: np.ndarray, start: float, end: float) -> np.ndarray:
    """sigmoid activation function with transition at x = 0
    Activation function to smoothly transition between start and end values.

    :param x: input array
    :param start: start value
    :param end: end value

    :return: sigmoid function
    """
    k = -0.4  # transition
    return start + (end - start) / (1 + np.exp(k * x))


def lerp(x: np.ndarray, start: float, end: float) -> np.ndarray:
    """linear interpolation between start and end values
    
    :param x: input array
    :param start: start value
    :param end: end value
    
    :return: linear interpolation
    """
    length = x[0, -1, 0] - x[0, 0, 0]
    t = (x + 0.5 * length) / length
    return start + (end - start) * t


def gaussian(x: np.ndarray, start: float, end: float) -> np.ndarray:
    """gaussian function with transition at x = 0

    :param x: input array
    :param start: start value
    :param end: end value

    :return: gaussian function
    """
    k = -0.02
    return start + (end - start) * np.exp(k * x**2)


def graded(
        x: np.ndarray, y: np.ndarray, z: np.ndarray,
        transition_func: Callable,
    ) -> np.ndarray:
    """graded cell size function
    
    :param x: input array
    :param y: input array
    :param z: input array
    
    :return: graded cell size function
    """
    min_cell_size = 2
    max_cell_size = 4
    # dim_x = dim_y = dim_z = lerp(x, min_cell_size, max_cell_size)
    # dim_x = dim_y = dim_z = sigmoid(x, min_cell_size, max_cell_size)
    # dim_x = dim_y = dim_z = gaussian(x, max_cell_size, min_cell_size)
    dim_x = dim_y = dim_z = transition_func(x, min_cell_size, max_cell_size)
    return gyroid(
        x * max_cell_size / dim_x,
        y * max_cell_size / dim_y,
        z * max_cell_size / dim_z,
    )


lerp_size = Tpms(
    surface_function=partial(graded, transition_func=lerp),
    offset=0.3,
    repeat_cell=(5, 1, 1),
    resolution=30,
)
lerp_size.sheet.extract_surface().clean().save("graded_cell_size.stl")

sigmoid_size = Tpms(
    surface_function=partial(graded, transition_func=sigmoid),
    offset=0.3,
    repeat_cell=(5, 1, 1),
    resolution=30,
)
sigmoid_size.sheet.extract_surface().clean().save("graded_cell_size_sigmoid.stl")

gaussian_size = Tpms(
    surface_function=partial(graded, transition_func=gaussian),
    offset=0.3,
    repeat_cell=(5, 1, 1),
    resolution=30,
)
gaussian_size.sheet.extract_surface().clean().save("graded_cell_size_gaussian.stl")

plotter = pv.Plotter(shape=(3, 1))

plotter.subplot(0, 0)
plotter.add_mesh(lerp_size.sheet, color="w")

plotter.subplot(1, 0)
plotter.add_mesh(sigmoid_size.sheet, color="w")

plotter.subplot(2, 0)
plotter.add_mesh(gaussian_size.sheet, color="w")

plotter.link_views()
plotter.view_xy()
plotter.enable_parallel_projection()
plotter.show_axes()
plotter.show()
