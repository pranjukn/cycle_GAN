import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.utils as vutils
from torchsummary import summary
from tqdm import tqdm
import itertools
import time
import os
from os.path import isfile, join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from google.colab import drive
from PIL import Image
import cv2
import random
import math
from collections import OrderedDict
from google.colab import drive
import sys
drive.mount('/content/drive')

def del_empty_pix(dir_A, dir_B, df):
  for files in os.listdir(dir_A):
    if None in df[files]['bboxes'][0]:
      os.remove(dir_A + '/' + files)
  for files in os.listdir(dir_B):
    if None in df[files]['bboxes'][0]:
      os.remove(dir_B + '/' + files)

def square_bboxes(bboxes):
    xmin = (bboxes[0])
    ymin = (bboxes[1])
    xmax = (bboxes[2])
    ymax = (bboxes[3])

    if (xmax - xmin) > (ymax - ymin):
      a = ymax
      b = ymin
      ymax = (a + b)/2 + (xmax - xmin)/2
      ymin = (a + b)/2 - (xmax - xmin)/2

    elif (ymax - ymin) > (xmax - xmin):
      a = xmax
      b = xmin
      xmax = (a + b)/2 + (ymax - ymin)/2
      xmin = (a + b)/2 - (ymax - ymin)/2

    if isinstance(xmax, int) == 0:
      xmax = int(math.ceil(xmax))
    if isinstance(xmin, int) == 0:
      xmin = int(math.ceil(xmin))
    if isinstance(ymax, int) == 0:
      ymax = int(math.ceil(ymax))
    if isinstance(ymin, int) == 0:
      ymin = int(math.ceil(ymin)) 

    return xmin, ymin, xmax, ymax

def gen_vid_out(pathIn, pathOut, df, fakes, fps):

  frame_array = []
  files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
  files.sort(key = lambda x: x[0:])
  files.sort()

  for i in range(len(files)):

    filename = pathIn + files[i]
    img = cv2.imread(filename)
    s_img = np.transpose(fakes[i].cpu(),(1,2,0)).cpu().detach().numpy()
    s_img -= s_img.min()
    s_img /= s_img.max()
    s_img = s_img * 255
    bboxes = df[files[i]]['bboxes'][0]
    xmin, ymin, xmax, ymax = square_bboxes(bboxes)  
    x = xmax - xmin
    y = ymax - ymin
    s_img = cv2.resize(s_img, dsize=(x, y), interpolation=cv2.INTER_CUBIC)
    x_offset = xmin
    y_offset = ymin
    img[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1]] = s_img
    height, width, layers = img.shape
    size = (width,height)
    frame_array.append(img)
    if i % 50 == 0:
      print(filename)
  out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'MP4V'), fps, size)
  for i in range(len(frame_array)):
    out.write(frame_array[i])
  print("Generated video up")
  out.release()


def og_vid_out(pathIn, pathOut, fps):
  frame_array = []
  files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
  files.sort(key = lambda x: x[0:])
  files.sort()
  for i in range(len(files)):
    filename = pathIn + files[i]
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    frame_array.append(img)
    if i % 50 == 0:
      print(filename)
  out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'MP4V'), fps, size)
  for i in range(len(frame_array)):
    out.write(frame_array[i])
  print("Original video up")
  out.release()

def loss_plot(loss_list):
  loss_D = []
  loss_G = []
  loss_cycle = []
  idt = []
  for i in range(0,len(loss_list)):
    loss_D.append(loss_list[i]['D_A'])
    loss_G.append(loss_list[i]['G_A'])
    loss_cycle.append(loss_list[i]['cycle_A'])
    idt.append(loss_list[i]['idt_A'])

  plt.figure(figsize=(10,5))
  plt.title("A domain networks losses during training")
  plt.plot(loss_D,label="D_A")
  plt.plot(loss_G,label="G_A")
  plt.plot(loss_cycle,label="cycle_A")
  plt.plot(idt,label="idt_A")
  plt.xlabel("iterations")
  plt.ylabel("Loss")
  plt.yscale('log')
  plt.legend()
  plt.show()

  loss_D = []
  loss_G = []
  loss_cycle = []
  idt = []
  for i in range(0,len(loss_list)):
    loss_D.append(loss_list[i]['D_B'])
    loss_G.append(loss_list[i]['G_B'])
    loss_cycle.append(loss_list[i]['cycle_B'])
    idt.append(loss_list[i]['idt_B'])

  plt.figure(figsize=(10,5))
  plt.title("B domain networks losses during training")
  plt.plot(loss_D,label="D_B")
  plt.plot(loss_G,label="G_B")
  plt.plot(loss_cycle,label="cycle_B")
  plt.plot(idt,label="idt_B")
  plt.xlabel("iterations")
  plt.ylabel("Loss")
  plt.yscale('log')
  plt.legend()
  plt.show()