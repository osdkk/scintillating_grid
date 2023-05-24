import os
import glob
import random

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2


class IllusionGridGenerator:
  def __init__(self, size, square_size, start_position, line_width, line_luminance=.5, circle_radius=0, circle_luminance=1.):
    self.size = size
    self.square_size = square_size
    self.start_position = start_position

    self.line_width = line_width
    self.line_luminance = line_luminance

    self.circle_radius = circle_radius
    self.circle_luminance = circle_luminance

    self.image = None
  
  def generate_grid(self):
    image = np.zeros((self.size, self.size), dtype=np.uint8)

    line_brightness = int(255 * self.line_luminance)
    circle_brightness = int(255 * self.circle_luminance)

    if self.line_width > 0:
      # 直線の描画（縦方向）
      for i in range(self.start_position, self.size, self.square_size):
        cv2.line(image, (0, i), (self.size, i), line_brightness, self.line_width)

      # 直線の描画（横方向）
      for j in range(self.start_position, self.size, self.square_size):
        cv2.line(image, (j, 0), (j, self.size), line_brightness, self.line_width)

    if self.circle_radius > 0:
      for i in range(self.start_position, self.size, self.square_size):
        for j in range(self.start_position, self.size, self.square_size):
          cv2.circle(image, (j, i), self.circle_radius, circle_brightness, -1)

    self.image = image
    return image
  
  def save_image(self, root_dir):
    if self.image is None:
      return 0
    
    if self.circle_radius > 0:
      # root/SG/円半径/正方形サイズ_開始位置_線幅/円輝度.png
      path = os.path.join(root_dir,
                          'SG',
                          str(self.circle_radius),
                          '{}_{}_{}'.format(self.square_size, self.start_position, self.line_width),
                          '{:.2f}.png'.format(self.circle_luminance))
    else:
      # root/HG/線幅/正方形サイズ_開始位置/線輝度.png
      path = os.path.join(root_dir,
                          'HG',
                          str(self.line_width),
                          '{}_{}'.format(self.square_size, self.start_position),
                          '{:.2f}.png'.format(self.line_luminance))
    
    if not os.path.exists(os.path.dirname(path)):
      os.makedirs(os.path.dirname(path))

    cv2.imwrite(path, self.image)