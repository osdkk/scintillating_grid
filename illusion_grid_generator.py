import os
import glob
import random

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2


class IllusionGridGenerator:
  def __init__(self,
               size,
               square_size,
               start_position,
               line_width,
               line_luminance=.5,
               curve_amplitude=0,
               circle_radius=0,
               circle_luminance=1.,
               offset_bars=False,
               scaling=8):
    
    self.size = size
    self.square_size = square_size
    self.start_position = start_position

    self.line_width = line_width
    self.line_luminance = line_luminance
    self.curve_amplitude = curve_amplitude

    self.circle_radius = circle_radius
    self.circle_luminance = circle_luminance
    self.offset_bars = offset_bars

    self.scaling = scaling
    self.image = None
  
  def generate_grid(self):
    self.image = np.zeros((self.size * self.scaling, self.size * self.scaling), dtype=np.uint8)

    line_brightness = int(255 * self.line_luminance)
    circle_brightness = int(255 * self.circle_luminance)

    if self.line_width > 0:
      # # 直線の描画（縦方向）
      # for i in range(self.start_position * self.scaling, self.size * self.scaling, self.square_size * self.scaling):
      #   cv2.line(self.image, (0, i), (self.size * self.scaling, i), line_brightness, self.line_width * self.scaling)

      # # 直線の描画（横方向）
      # for j in range(self.start_position * self.scaling, self.size * self.scaling, self.square_size * self.scaling):
      #   cv2.line(self.image, (j, 0), (j, self.size * self.scaling), line_brightness, self.line_width * self.scaling)

      line = np.linspace(0, self.size * self.scaling, self.size * self.scaling+1, dtype=np.int32)
      for i in range(self.start_position * self.scaling, self.size * self.scaling, self.square_size * self.scaling):
        sin_curve = i + self.curve_amplitude * self.scaling * np.sin( (line-self.start_position * self.scaling) / (self.square_size * self.scaling / (2 * np.pi)))
        sin_curve = np.round(sin_curve).astype(int)
        for j in range(sin_curve.shape[0] - 1):
          cv2.line(self.image, (line[j], sin_curve[j]), (line[j + 1], sin_curve[j + 1]), line_brightness, self.line_width * self.scaling)
          cv2.line(self.image, (sin_curve[j], line[j]), (sin_curve[j + 1], line[j + 1]), line_brightness, self.line_width * self.scaling)

    if self.circle_radius > 0:
      circle_start_postion = self.square_size // 2 if self.offset_bars else 0
      for i in range((self.start_position + circle_start_postion) * self.scaling, self.size * self.scaling, self.square_size * self.scaling):
        for j in range((self.start_position + circle_start_postion)  * self.scaling, self.size * self.scaling, self.square_size * self.scaling):
          cv2.circle(self.image, (j, i), self.circle_radius * self.scaling, circle_brightness, -1)

    self.image = cv2.resize(self.image, (self.size, self.size), interpolation=cv2.INTER_CUBIC)
    return self.image
  
  def save_image(self, root_dir, type_name):
    if self.image is None:
      return 0
    
    # root/type/正方形サイズ_開始位置/円輝度.png
    path = os.path.join(root_dir,
                        type_name,
                        '{}_{}'.format(self.square_size, self.start_position),
                        '{:.2f}.png'.format(self.circle_luminance))

    # if self.circle_radius > 0:
    #   # root/SG/円半径/線幅/正方形サイズ_開始位置/円輝度.png
    #   path = os.path.join(root_dir,
    #                       'SG',
    #                       str(self.circle_radius),
    #                       str(self.line_width),
    #                       '{}_{}'.format(self.square_size, self.start_position),
    #                       '{:.2f}.png'.format(self.circle_luminance))
    # else:
    #   # root/HG/線幅/正方形サイズ_開始位置/線輝度.png
    #   path = os.path.join(root_dir,
    #                       'HG',
    #                       str(self.line_width),
    #                       '{}_{}'.format(self.square_size, self.start_position),
    #                       '{:.2f}.png'.format(self.line_luminance))
    
    if not os.path.exists(os.path.dirname(path)):
      os.makedirs(os.path.dirname(path))

    cv2.imwrite(path, self.image)