import sys
import argparse


import numpy as np
import tensorflow as tf

from n2v.models import N2VConfig, N2V

print('everything imported')
import numpy as np
from csbdeep.utils import plot_history
from n2v.utils.n2v_utils import manipulate_val_data
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
from matplotlib import pyplot as plt
import urllib
import os
import zipfile
import torch


base_path = "D:\\wlz\\code\\n2v-main\\n2v-main\\data\\lixufanhuadata\\train\\1"
datagen = N2V_DataGenerator()
imgs = datagen.load_imgs_from_directory(directory = base_path, dims='YX', filter="1DD02.tif")
print("imgs.shape",imgs[0].shape)