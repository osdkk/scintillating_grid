import os
import glob
import random

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

from illusion_grid_generator import IllusionGridGenerator


grid_params = {
  'size': 224,
  'square_size': None,
  'start_position': None,

  'line_width': 4,
  'line_luminance': .5,
  'curve_amplitude': 0,
  
  'circle_radius': 5,
  'circle_luminance': 1.,
  'offset_bars': False,
  
  'scaling': 4,
}

grid_type = {
  'a_scintillating': grid_params.copy(),
  'b_sin_curve': grid_params.copy(),
  'c_large_circle': grid_params.copy(),
  'd_offset_bars': grid_params.copy(),
  'e_no_bars': grid_params.copy(),
}

grid_type['b_sin_curve']['curve_amplitude'] = 4
grid_type['c_large_circle']['circle_radius'] = 10
grid_type['d_offset_bars']['offset_bars'] = True
grid_type['e_no_bars']['line_width'] = 0


data_dir = './data'


if __name__ == '__main__':
  square_size_list = np.arange(40, 120, 5, dtype=np.int32)
  start_position_list = np.arange(10, 90, 10, dtype=np.int32)

  mu_list = np.arange(0, 1.05, 0.05)

  for type_name in grid_type.keys():
    grid = IllusionGridGenerator(**grid_type[type_name])

    for square_size in square_size_list:
      grid.square_size = square_size
      for start_position in start_position_list:
        grid.start_position = start_position

        for mu in mu_list:
          grid.circle_luminance = mu

          grid.generate_grid()
          grid.save_image(data_dir, type_name)