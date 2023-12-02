import os
import numpy as np
import nibabel as nib
from scipy import ndimage
import SimpleITK as sitk
import six
import os  # needed navigate the system to get the input data
import numpy as np
import radiomics
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor  # This module is used for interaction with pyradiomics
import argparse

def __len_file__(file):
    return(len(file))

def get_bbox_3d(mask, ratio=0.5):
    d, w, h= mask.shape
    zz, xx, yy = np.where(mask >0)
    x_min, x_max = np.min(xx), np.max(xx)
    y_min, y_max = np.min(yy), np.max(yy)
    z_min, z_max = np.min(zz), np.max(zz)
    width, height, depth = x_max - x_min, y_max - y_min, z_max - z_min
    x_min, x_max = max(0, int(x_min - (ratio * width) //2)), min(int(x_max + (ratio * width) //2), w)
    y_min, y_max = max(0,int(y_min - (ratio * height) //2)), min(int(y_max + (ratio * height) //2),h)
    return [x_min, y_min, x_max, y_max, z_min, z_max]

def correct_bias(in_file, out_file, image_type=sitk.sitkFloat64):
    """
    Corrects the bias using SimpleITK N4BiasFieldCorrection.
    :param in_file: .nii.gz 文件的输入路径
    :param out_file: .nii.gz 校正后的文件保存路
    :return: 校正后的nii文件全路径名
    """
    # 使用SimpltITK N4BiasFieldCorrection校正MRI图像的偏置场
    input_image = sitk.ReadImage(in_file, image_type)
    output_image_s = sitk.N4BiasFieldCorrection(input_image, input_image > 0)
    sitk.WriteImage(output_image_s, out_file)
    return os.path.abspath(out_file)

def read_nii(img):
    img_new = nib.load(img).get_fdata()
    return(img_new)

def save_nii(img_arr, save_dir):
    img_ = sitk.GetImageFromArray(img_arr)
    sitk.WriteImage(img_, save_dir)

def normalize(volume):
    max_value = np.percentile(volume,99) + 0.01
    min_value = 0
    volume[volume>max_value] = max_value
    volume[volume<min_value] = min_value
    volume = volume/max_value
    return volume


def normalize_3d(volume):
    max_value = volume.max()
    min_value = 0
    volume[volume>max_value] = max_value
    volume[volume<min_value] = min_value
    volume = volume/max_value
    return volume


# sitk.sitkNearestNeighbor mask用这个插值 最近邻

def resize_volume(img,desired_width,desired_height,desired_depth, order = 1):  # 所需深度 宽度 高度
    """跨 z 轴调整大小"""
    # 获取当前深度
    current_width,current_height, current_depth = img.shape
    # 计算深度因子
    width = current_width / desired_width
    height = current_height / desired_height
    depth = current_depth / desired_depth
    width_factor = 1 / width
    height_factor = 1 / height
    depth_factor = 1 / depth
    # 旋转
#     img = ndimage.rotate(img, 90, reshape=False)
    # 跨z轴调整
    img = ndimage.zoom(img,(width_factor, height_factor, depth_factor),order = order)
    return img
# order 参数的取值 0 表示最近邻插值，1 表示线性插值，2 表示二次插值，3 表示三次插值。



def get_tumor_3d(T1c ,mask,ratio=6, type = True):
    T1c_arr = read_nii(T1c)
    mask_arr = read_nii(mask)
    bbox = get_bbox_3d(mask_arr,ratio=ratio)
    img = T1c_arr[bbox[0]: bbox[2], bbox[1]: bbox[3], bbox[4]: bbox[5]]
    mask_ = mask_arr[bbox[0]: bbox[2], bbox[1]: bbox[3], bbox[4]: bbox[5]]
    return img, mask_
   









