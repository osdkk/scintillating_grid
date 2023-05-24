import os
import glob
import random
import argparse
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import cv2

import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import vgg19, VGG19_Weights


parser = argparse.ArgumentParser(description='Representation Dissimilarity')
parser.add_argument('--data_dir', '-d', default='./data/SG/4/')


args = parser.parse_args()


def fix_seed(seed, determisnistic=False):
  random.seed(seed)
  np.random.seed(seed)

  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  if determisnistic:
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

class RepresentationalDissimilarity:
  def __init__(self):
    self.r_L1 = None
    self.labels = []
  
  def register_r_L1(self, representation, label, ref_index=-1):
    self.labels.append(label)

    with torch.inference_mode():
      ref = representation[ref_index].unsqueeze(0).repeat(11, 1)[0]
      r = torch.sum(torch.abs(representation - ref), dim=1).unsqueeze(0)

      if self.r_L1 is None:
        self.r_L1 = r.clone()
      else:
        self.r_L1 = torch.concat([self.r_L1, r.clone()], dim=0)    

  def R(self):
    mean_r_L1 = self.r_L1.mean()
    return self.r_L1 / mean_r_L1
  
  def plot_R(self, root_dir, title):
    R = self.R().cpu().clone().numpy()

    x = np.linspace(0, 1, R.shape[1])
    R_mean = np.mean(R, axis=0)

    alpha = 0.3
    shade_min = np.percentile(R, 25, axis=0)
    shade_max = np.percentile(R, 75, axis=0)

    fig, ax = plt.subplots()
    ax.plot(x, R_mean, color='black', linewidth=2)  # 平均線のプロット
    ax.fill_between(x, shade_min, shade_max, color='gray', alpha=alpha)  # 分散の範囲をグレーで塗りつぶし
    ax.set_title(title)
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 2.5))
    ax.invert_xaxis()

    if not os.path.exists(root_dir):
      os.makedirs(root_dir)
    
    path = os.path.join(root_dir, '{}.png'.format(title))
    fig.savefig(path)

def generate_save_title(data_dir):
  dirs = data_dir.split('/')
  try:
    mode = 'HG'
    idx = dirs.index(mode)
    save_title = '{}_{}'.format(mode, dirs[idx+1])
  except ValueError as e:
    mode = 'SG'
    idx = dirs.index(mode)
    save_title = '{}_{}'.format(mode, dirs[idx+1])
  return save_title



if __name__ == '__main__':
  seed = 42
  data_dir = args.data_dir

  fix_seed(seed, determisnistic=True)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print('device: {}'.format(device))
  
  model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).to(device)
  
  transform = T.Compose([
    T.PILToTensor(),
    VGG19_Weights.IMAGENET1K_V1.transforms()
  ])
  dataset = ImageFolder(root=data_dir, transform=transform)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=21, shuffle=False, num_workers=1, pin_memory=True)
  
  r = RepresentationalDissimilarity()

  model.eval()
  with torch.inference_mode():
    for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
      x, t = map(lambda x: x.to(device), data)
      assert t.size()[0] == (t == i).sum()

      fc_8 = model(x)
      r.register_r_L1(fc_8, dataset.classes[i])

  save_title = generate_save_title(data_dir)
  r.plot_R('./result/', save_title)