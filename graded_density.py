import numpy as np
import pyvista as pv

from microgen import Tpms
from microgen.shape.surface_functions import gyroid

def linear_graded_density(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    length = y[-1, :, :] - y[0, :, :]
    return min_offset + (max_offset - min_offset) * (y + 0.5 * length) / length 

def circular_graded_density(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    radius = 2.3
    return min_offset + (max_offset - min_offset) * (x**2 + y**2) / radius**2

min_offset = 0.5
max_offset = 3.0

circular = Tpms(
    surface_function=gyroid,
    offset=circular_graded_density,
    repeat_cell=(5, 5, 1),
    cell_size=(1, 1, 1),
    resolution=30,
)

linear = Tpms(
    surface_function=gyroid,
    offset=linear_graded_density,
    repeat_cell=(1, 5, 1),
    cell_size=(1, 1, 1),
    resolution=30,
)
linear_sheet = linear.sheet
linear_sheet.extract_surface().clean().save("linear_graded_density.stl")
circular_sheet = circular.sheet
circular_sheet.extract_surface().clean().save("circular_graded_density.stl")

plotter = pv.Plotter(shape=(1, 2))

plotter.subplot(0, 0)
plotter.add_mesh(linear_sheet, color="w")
plotter.view_xy()
plotter.enable_parallel_projection()
plotter.show_axes()

plotter.subplot(0, 1)
plotter.add_mesh(circular_sheet, color="w")
plotter.view_xy()
plotter.enable_parallel_projection()
plotter.show_axes()

plotter.show()
