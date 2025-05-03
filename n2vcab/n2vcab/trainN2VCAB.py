#!/usr/bin/env python3

import os
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

from tifffile import imread
from tifffile import imwrite

import glob

# 设置随机数种子
torch.manual_seed(123)  # 为CPU设置随机种子
torch.cuda.manual_seed(123)  # 为当前GPU设置随机种子
#seed_value = 42
#os.environ['PYTHONHASHSEED']=str(seed_value)
#np.random.seed(seed_value)
#tf.random.set_seed(seed_value)

# 在训练代码中加入随机数种子
# ...


def list_numbered_subdirectories(base_path, start_number, end_number):
    subdirectories = []

    for i in range(start_number, end_number + 1):
        subdir_path = os.path.join(base_path, str(i))
        subdirectories.append(subdir_path)

    return subdirectories


    # 替换为你的基础路径
base_path = "D:\\wlz\\code\\n2v-main\\n2v-main\\data\\240403\\train"

# 定义范围，例如1到10
start_number =1
end_number =12
    # 获取1到10文件夹路径
numbered_subdirectories = list_numbered_subdirectories(base_path, start_number, end_number)

    # 打印每个文件夹路径
for subdir in numbered_subdirectories:
    print(subdir)

model_names = ['1', '2', '3','4','5','6','7','8','9','10','11','12']#,'13','14','15','16','17','18']#,'19','20', '21', '22','23','24','25','26','27']
#'1', '2', '3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20', '21', '22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','eeee
for models in model_names:
    print(models)

# 在strategy范围内构建模型和训练过
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--baseDir", help="base directory in which your network will live", default='models')
parser.add_argument("--name", help="name of your network", default='N2V')
parser.add_argument("--dataPath", help="The path to your training data")
parser.add_argument("--fileName", help="name of your training data file", default="*.tif")
parser.add_argument("--validationFraction", help="Fraction of data you want to use for validation (percent)",default=5.0, type=float)
parser.add_argument("--dims", help="dimensions of your data, can include: X,Y,Z,C (channel), T (time)", default='YX')
parser.add_argument("--patchSizeXY", help="XY-size of your training patches", default=32, type=int)
parser.add_argument("--patchSizeZ", help="Z-size of your training patches", default=32, type=int)
parser.add_argument("--epochs", help="number of training epochs", default=70, type=int)
parser.add_argument("--stepsPerEpoch", help="number training steps per epoch", default=400, type=int)
parser.add_argument("--batchSize", help="size of your training batches", default=32, type=int)
parser.add_argument("--netDepth", help="depth of your U-Net", default=2, type=int)
parser.add_argument("--netKernelSize", help="Size of conv. kernels in first layer", default=3, type=int)
parser.add_argument("--n2vPercPix", help="percentage of pixels to manipulated by N2V", default=1.6, type=float)
parser.add_argument("--learningRate", help="initial learning rate", default=0.0004, type=float)
parser.add_argument("--unet_n_first", help="number of feature channels in the first u-net layer", default=32,type=int)
parser.add_argument("--noAugment", action='store_true', help="do not rotate and flip training patches")


for datapath,modelname in zip(numbered_subdirectories,model_names):



    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    print(args)




    args.dataPath=datapath
    args.baseDir=modelname
    print(datapath)
    print(args.dataPath)
    print(modelname)



    print('everything imported')


    print("args",str(args.name))

    print('augment',(not args.noAugment))



    ####################################################
    #           PREPARE TRAINING DATA
    ####################################################


    datagen = N2V_DataGenerator()
    imgs = datagen.load_imgs_from_directory(directory = args.dataPath, dims=args.dims, filter=args.fileName)
    print("imgs.shape",imgs[0].shape)

    # Here we extract patches for training and validation.
    pshape=( args.patchSizeXY, args.patchSizeXY)
    if 'Z' in args.dims:
        pshape=(args.patchSizeZ, args.patchSizeXY, args.patchSizeXY)

    print(pshape)
    patches = datagen.generate_patches_from_list(imgs[:1], shape=pshape, augment=(not args.noAugment))

    # The patches are non-overlapping, so we can split them into train and validation data.
    frac= int( (len(patches))*float(args.validationFraction)/100.0)
    print("total no. of patches: "+str(len(patches)) + "\ttraining patches: "+str(len(patches)-frac)+"\tvalidation patches: "+str(frac))
    X = patches[frac:]
    X_val = patches[:frac]



    config = N2VConfig(X, unet_kern_size=args.netKernelSize,
                       train_steps_per_epoch=int(args.stepsPerEpoch),train_epochs=int(args.epochs), train_loss='mse', batch_norm=True,
                       train_batch_size=args.batchSize, n2v_perc_pix=args.n2vPercPix, n2v_patch_shape=pshape,
                       n2v_manipulator='uniform_withCP', n2v_neighborhood_radius=5, train_learning_rate=args.learningRate,
                       unet_n_depth=args.netDepth,
                       unet_n_first=args.unet_n_first
                       )

    # Let's look at the parameters stored in the config-object.
    vars(config)


    # a name used to identify the model
    model_name = args.name
    # the base directory in which our model will live
    basedir = args.baseDir
    # We are now creating our network model.
    datapath=args.dataPath
    model = N2V(config=config, name=model_name, basedir=basedir)



    ####################################################
    #           Train Network
    ####################################################
    print("begin training")
    history = model.train(X, X_val)

