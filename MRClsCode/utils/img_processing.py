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

def convert_path(path_dir):
    converted_path = path_dir.replace('\\', '/')
    return converted_path

def name_pinyin(x):
    number = len(x)
    p = Pinyin()
    result = p.get_pinyin(x)
    s = result.split('-')
    if number == 3:
        name = s[0].capitalize()+'_'+'_'.join(s[1:2]).capitalize()+'_'+'_'.join(s[2:]).capitalize()
    elif number == 2:
        name = s[0].capitalize()+'_'+'_'.join(s[1:]).capitalize()
    else:
        name = x
    return name

def zip_folder(folder_path, output_path):
    zipf = zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED)
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            zipf.write(os.path.join(root, file))
    zipf.close()

def __len_file__(file):
    return(len(file))

def gaussian_prob(data,avg,sig):
    sqrt_2pi=np.power(2*np.pi,0.5)
    coef=1/(sqrt_2pi*sig)
    powercoef=-1/(2*np.power(sig,2))
    mypow=powercoef*(np.power((data-avg),2))
    prob = coef*(np.exp(mypow))
    return prob
'''
示例
>> x=np.arange(0,1.1,0.1)
>> p=prob(x,0.3,0.2)
output: p
array([0.64758798, 1.20985362, 1.76032663, 1.9947114 , 1.76032663,
       1.20985362, 0.64758798, 0.26995483, 0.0876415 , 0.02215924,
       0.00436341])
''' 

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print(path)
    else:
        print(path,'is this folder')

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

def resample_image(img_path, newSpacing=[1.0,1.0,1.0], resamplemethod=sitk.sitkLinear):
    image = sitk.ReadImage(img_path)
    print('原始图像的Spacing: ', image.GetSpacing())
    print('原始图像的Size: ', image.GetSize())
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(resamplemethod)
    resample.SetOutputDirection(image.GetDirection()) ##
    resample.SetOutputOrigin(image.GetOrigin()) ##

    newSpacing = np.array(newSpacing, float)
    newSize = image.GetSize() / newSpacing * image.GetSpacing()
    newSize = np.around(newSize,decimals=0)
    newSize = newSize.astype(np.int64)

    resample.SetSize(newSize.tolist()) ##
    resample.SetOutputSpacing(newSpacing) ##
    Resample_img = resample.Execute(image)
    print('经过resample之后图像的Spacing是: ', Resample_img.GetSpacing())
    print('经过resample之后图像的Size是: ', Resample_img.GetSize())
    Resample_arr = sitk.GetArrayFromImage(Resample_img)
    return Resample_arr #返回sitk类型的数据矩阵

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

def save_nii(img_arr, save_dir):
    img_ = sitk.GetImageFromArray(img_arr)
    sitk.WriteImage(img_, save_dir)

def resize_volume(img,desired_width,desired_height,order = 1):  # 所需深度 宽度 高度
    """跨 z 轴调整大小"""
    # 获取当前深度
    current_width,current_height = img.shape
    # 计算深度因子
    width = current_width / desired_width
    height = current_height / desired_height
    width_factor = 1 / width
    height_factor = 1 / height
    # 旋转
#     img = ndimage.rotate(img, 90, reshape=False)
    # 跨z轴调整
    img = ndimage.zoom(img,(width_factor, height_factor),order = order)
    return img
# order 参数的取值 0 表示最近邻插值，1 表示线性插值，2 表示二次插值，3 表示三次插值。

def get_tumor_core(T1c_arr,bbox,order =1):
    img = T1c_arr[bbox[0]:bbox[2],bbox[1]:bbox[3]]
    img_resize = resize_volume(img,128,128,order=order)
    return img_resize[np.newaxis, ...]

def get_bbox(mask, ratio):
    w, h = mask.shape
    xx, yy = np.where(mask >0)
    x_min, x_max = np.min(xx), np.max(xx)
    y_min, y_max = np.min(yy), np.max(yy)
    width, height = x_max - x_min, y_max - y_min
    x_min, x_max = max(0, int(x_min - (ratio * width) //2)), min(int(x_max + (ratio * width) //2), w)
    y_min, y_max = max(0,int(y_min - (ratio * height) //2)), min(int(y_max + (ratio * height) //2),h)
    return [x_min, y_min, x_max, y_max]


def get_tumor(T1c ,mask,ratio=6, type = True):
    T1c_arr = read_nii(T1c)
    mask_arr = read_nii(mask)
    x_mask,y_mask,z_mask = mask_arr.shape
    arr = np.empty((0,1),int)
    for i in range(z_mask):
        slice_num = np.sum(mask_arr[:,:,i])
        arr = np.append(arr,np.array(slice_num))
    max_roi = np.argmax(arr)
    bbox = get_bbox(mask_arr[:,:,max_roi],ratio=ratio)
    img_core_1 = normalize(get_tumor_core(T1c_arr[:,:,max_roi-1],bbox))
    mask_core_1 = get_tumor_core(mask_arr[:,:,max_roi-1],bbox,order=0)
    img_core_2 = normalize(get_tumor_core(T1c_arr[:,:,max_roi],bbox))
    mask_core_2 = get_tumor_core(mask_arr[:,:,max_roi],bbox,order=0)
    img_core_3 = normalize(get_tumor_core(T1c_arr[:,:,max_roi+1],bbox))
    mask_core_3 = get_tumor_core(mask_arr[:,:,max_roi+1],bbox,order=0)
    if type:
        img_core = np.concatenate((img_core_1,mask_core_1, img_core_2,mask_core_2,img_core_3, mask_core_3),axis = 0) # 标准化和resize
    else:
        img_core =  np.concatenate((img_core_1,img_core_2, img_core_3),axis = 0)
    
    return img_core
   

def save_npy(save_dir,file):
    file = np.array(file)
    np.save(save_dir,file)


def split_train_test(img_arr, mask_arr, label_arr, random_seed = 2020, ratio = 0.8):
    np.random.seed(random_seed)
    random_index = np.random.permutation(len(label_arr))
    mask_arr_ = np.array(mask_arr)[random_index]
    img_arr_ = np.array(img_arr)[random_index]
    label_arr_ = np.array(label_arr)[random_index]
    i = int(len(label_arr)*0.8)
    train_img = img_arr_[:i]
    train_mask = mask_arr_[:i]
    train_status = label_arr_[:i]
    test_img = img_arr_[i:]
    test_mask = mask_arr_[i:]
    test_status = label_arr_[i:]
    return train_img, train_mask, train_status, test_img, test_mask, test_status



def get_tumor_seg(T1c ,mask,ratio):
    T1c_arr = read_nii(T1c)
    mask_arr = read_nii(mask)
    x_mask,y_mask,z_mask = mask_arr.shape
    arr = np.empty((0,1),int)
    for i in range(z_mask):
        slice_num = np.sum(mask_arr[:,:,i])
        arr = np.append(arr,np.array(slice_num))
    max_roi = np.argmax(arr)
    bbox = get_bbox(mask_arr[:,:,max_roi],ratio=ratio)
    img_core_1 = get_tumor_core(T1c_arr[:,:,max_roi-1],bbox)
    mask_core_1 = get_tumor_core(mask_arr[:,:,max_roi-1],bbox,order=0)
    img_core_2 = get_tumor_core(T1c_arr[:,:,max_roi],bbox)
    mask_core_2 = get_tumor_core(mask_arr[:,:,max_roi],bbox,order=0)
    img_core_3 = get_tumor_core(T1c_arr[:,:,max_roi+1],bbox)
    mask_core_3 = get_tumor_core(mask_arr[:,:,max_roi+1],bbox,order=0)
    img_core =normalize(np.concatenate((img_core_1,img_core_2,img_core_3),axis = 0)) # 标准化和resize
    mask_core = np.concatenate((mask_core_1,mask_core_2,mask_core_3),axis = 0)
    return img_core,mask_core



# def get_tumor_seg_max(T1c ,mask,ratio):
#     T1c_arr = read_nii(T1c)
#     mask_arr = read_nii(mask)
#     x_mask,y_mask,z_mask = mask_arr.shape
#     arr = np.empty((0,1),int)
#     for i in range(z_mask):
#         slice_num = np.sum(mask_arr[:,:,i])
#         arr = np.append(arr,np.array(slice_num))
#     max_roi = np.argmax(arr)
#     bbox = get_bbox(mask_arr[:,:,max_roi],ratio=ratio)
#     img_core_1 = normalize(get_tumor_core(T1c_arr[:,:,max_roi],bbox))
#     mask_core_1 = get_tumor_core(mask_arr[:,:,max_roi],bbox,order=0)
#     mask_resize = mask_core_1.squeeze(0)
#     return img_core_1,mask_resize


def get_tumor_seg_max(img_arr,mask_arr,ratio):
    max_roi = 0
    z_mask = mask_arr.shape[0]
    arr = np.empty((0,1),int)
    for i in range(z_mask):
        slice_num = np.sum(mask_arr[i,:,:])
        arr = np.append(arr,np.array(slice_num))
    max_roi = np.argmax(arr)
    bbox = get_bbox(mask_arr[max_roi,:,:],ratio=ratio)
    for i in range(img_arr.shape[0]):
        img_core = normalize(get_tumor_core(img_arr[i,max_roi,:,:],bbox))
        if i == 0:
            img_core_ = img_core
        else:
            img_core_ = np.concatenate((img_core_, img_core),axis = 0)
    mask_core = get_tumor_core(mask_arr[max_roi,:,:],bbox,order=0)
    return img_core_, mask_core

def read_npy(file_dir):
    file = np.load(file_dir)
    return file


