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
import zipfile
from xpinyin import Pinyin
from sklearn.preprocessing import MinMaxScaler

def normalize_min_max(df,norm_col = 'score',type = False):
    X_processing =df[norm_col].values.reshape(-1, 1)
    max_value, min_value = X_processing.max(),X_processing.min()
    scaler = MinMaxScaler()
    scaler.fit(X_processing)
    X_processing1 = scaler.transform(X_processing)
    df[norm_col] = X_processing1
    if type:
        return max_value,min_value,df
    else:
        return df


def __len_file__(file):
    return(len(file))


def read_nii(img):
    img_new = nib.load(img).get_fdata()
    return(img_new)


def normalize(volume):
    max_value = np.percentile(volume,99) + 0.01
    min_value = 0
    volume[volume>max_value] = max_value
    volume[volume<min_value] = min_value
    volume = volume/max_value
    return volume

def catch_features(imagePath,maskPath):
    if imagePath is None or maskPath is None:  # Something went wrong, in this case PyRadiomics will also log an error
        raise Exception('Error getting testcase!')  # Raise exception to prevent cells below from running in case of "run all"
    settings = {}
    settings['binWidth'] = 25  # 5
    settings['sigma'] = [3, 5]
    settings['Interpolator'] = sitk.sitkBSpline
    settings['resampledPixelSpacing'] = [1, 1, 1]  # 3,3,3
    settings['voxelArrayShift'] = 1000  # 300
    settings['normalize'] = True
    settings['normalizeScale'] = 100
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    #extractor = featureextractor.RadiomicsFeatureExtractor()
    print('Extraction parameters:\n\t', extractor.settings)
    extractor.enableImageTypeByName('LoG')
    extractor.enableImageTypeByName('Wavelet')
    extractor.enableImageTypeByName('Square')
    extractor.enableImageTypeByName('Exponential')
    extractor.enableImageTypeByName('SquareRoot')
    extractor.enableImageTypeByName('Logarithm')
    extractor.enableAllFeatures()
    extractor.enableFeaturesByName(firstorder=['Energy', 'TotalEnergy', 'Entropy', 'Minimum', '10Percentile', '90Percentile', 'Maximum', 'Mean', 'Median', 'InterquartileRange', 'Range', 'MeanAbsoluteDeviation', 'RobustMeanAbsoluteDeviation', 'RootMeanSquared', 'StandardDeviation', 'Skewness', 'Kurtosis', 'Variance', 'Uniformity'])
    extractor.enableFeaturesByName(shape=['VoxelVolume', 'MeshVolume', 'SurfaceArea', 'SurfaceVolumeRatio', 'Compactness1', 'Compactness2', 'Sphericity', 'SphericalDisproportion','Maximum3DDiameter','Maximum2DDiameterSlice','Maximum2DDiameterColumn','Maximum2DDiameterRow', 'MajorAxisLength', 'MinorAxisLength', 'LeastAxisLength', 'Elongation', 'Flatness'])
    # 上边两句我将一阶特征和形状特征中的默认禁用的特征都手动启用，为了之后特征筛选
    print('Enabled filters:\n\t', extractor.enabledImagetypes)
    feature_cur = []
    feature_name = []
    result = extractor.execute(imagePath, maskPath, label=1)
    for key, value in six.iteritems(result):
        print('\t', key, ':', value)
        feature_name.append(key)
        feature_cur.append(value)
    print(len(feature_cur[37:]))
    name = feature_name[37:]
    name = np.array(name)
    for i in range(len(feature_cur[37:])):
        #if type(feature_cur[i+22]) != type(feature_cur[30]):
        feature_cur[i+37] = float(feature_cur[i+37])
    return feature_cur[37:],name

   

def save_npy(save_dir,file):
    file = np.array(file)
    np.save(save_dir,file)




