#coding:utf-8

# 首先对三维脊柱进行竖直化处理
# 并且对每个椎体分别单独处理切除其椎弓根
# 截取三维区域并且保存为nii.gz
# 保存大小为256 256 64，刚好可以囊括椎体并且看到周围椎体

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import os
import numpy as np
import nibabel as nib
from PIL import Image
from skimage import morphology
from skimage.transform import resize
import cv2
import os
import numpy as np
import nibabel as nib
from PIL import Image
import matplotlib.pyplot as plt
from skimage import measure
import skimage
import json
from scipy.ndimage import label as Label
from scipy.ndimage import map_coordinates
from straighten import Interpolator
from scipy.optimize import curve_fit
import nibabel.orientations as nio

def find_largest_file(folder_path):
    largest_file_path = None
    largest_file_size = 0

    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            file_size = os.path.getsize(file_path)
            if file_size > largest_file_size:
                largest_file_path = file_path
                largest_file_size = file_size

    return largest_file_path

def reorient_to(img, axcodes_to=('R', 'A', 'I'), verb=False):
    aff = img.affine
    arr = np.asanyarray(img.dataobj, dtype=img.dataobj.dtype)
    ornt_fr = nio.io_orientation(aff)
    ornt_to = nio.axcodes2ornt(axcodes_to)
    ornt_trans = nio.ornt_transform(ornt_fr, ornt_to)
    arr = nio.apply_orientation(arr, ornt_trans)
    aff_trans = nio.inv_ornt_aff(ornt_trans, arr.shape)
    newaff = np.matmul(aff, aff_trans)
    newimg = nib.Nifti1Image(arr, newaff)
    if verb:
        print("[*] Image reoriented from", nio.ornt2axcodes(ornt_fr), "to", axcodes_to)
    return newimg

def poly_func(z, a, b, c):
    return a * z**2 + b * z + c

# 目的是为了延长曲线的两点，不然生成的图像只到最上层的椎体和最下层椎体的中心位置，椎体被截断了
# 效果不佳
def extend_coordinates(coordinates,zmin,zmax):
    
    # 获取 z 坐标和对应的 x, y 坐标
    z_coords = coordinates[:, 0]
    x_coords = coordinates[:, 1]
    y_coords = coordinates[:, 2]

    # 对 x(z) 和 y(z) 进行拟合
    params_x, _ = curve_fit(poly_func, z_coords, x_coords)
    params_y, _ = curve_fit(poly_func, z_coords, y_coords)

    # 预测 z 轴两端延长 20 像素后的新坐标
    z_min_new = max(z_coords.min() - 20,zmin)
    z_max_new = min(z_coords.max() + 20,zmax)
    new_x_min = poly_func(z_min_new, *params_x)
    new_x_max = poly_func(z_max_new, *params_x)
    new_y_min = poly_func(z_min_new, *params_y)
    new_y_max = poly_func(z_max_new, *params_y)

    # 构建新的坐标点并添加到数组头尾
    new_first_point = [z_min_new, new_x_min, new_y_min]
    new_last_point = [z_max_new, new_x_max, new_y_max]
    extended_coordinates = np.vstack([new_first_point, coordinates, new_last_point])
    
    return extended_coordinates

def clamp(value, min_value, max_value):
    """辅助函数，用于将值限制在给定的最小值和最大值之间"""
    return max(min_value, min(max_value, value))

# 直接使用曲线两端的两个点计算延长线应该是最有效的！
# 延长距离设定为20是比较合适的值
def extend_curve(curve, extension_length, min_bounds, max_bounds):
    """
    curve: numpy数组，形状为(n, 3)，每行代表曲线上的一个点(z, x, y)
    extension_length: 延长的距离
    min_bounds: 最小边界(z_min, x_min, y_min)
    max_bounds: 最大边界(z_max, x_max, y_max)
    """
    
    # 向曲线结束方向延长
    direction_end = curve[-1] - curve[-2]
    direction_end_normalized = direction_end / np.linalg.norm(direction_end)
    new_point_end = curve[-1] + direction_end_normalized * extension_length
    # 边界控制
    new_point_end = np.array([clamp(new_point_end[i], min_bounds[i], max_bounds[i]) for i in range(3)])
    
    # 向曲线开始方向延长
    direction_start = curve[0] - curve[1]
    direction_start_normalized = direction_start / np.linalg.norm(direction_start)
    new_point_start = curve[0] + direction_start_normalized * extension_length
    # 边界控制
    new_point_start = np.array([clamp(new_point_start[i], min_bounds[i], max_bounds[i]) for i in range(3)])
    
    # 更新曲线
    extended_curve = np.vstack([new_point_start, curve, new_point_end])
    
    return extended_curve

def remove_spine_labels_after_split(label_image):
    # 获取图像的维度信息
    depth, height, width = label_image.shape
    #print(label_image.shape)
    # 计算第二维（y轴）的中心位置
    center_y = height // 2
    # 遍历每一个椎体值
    unique_labels = np.unique(label_image)
    unique_labels = unique_labels[unique_labels != 0]  # 排除背景标签（假设为0）
    
    for label in unique_labels:
        # 从中心层向后遍历每一层
        for h in range(center_y,height):
            # 检查该层中心线上是否存在当前椎体的标签
            if label not in label_image[:, h, width // 2]:
                #print(h)
                # 从这一层开始，去除之后层的该椎体标签
                # 使用布尔数组直接索引需要修改的层
                #for z in range(h, height):
                label_image[:,h:height,:][label_image[:,h:height,:] == label] = 0
                break  # 处理完当前椎体后跳出循环，继续下一个椎体

    return label_image

# 简单处理效果也不行
def extend_coordinates(coordinates,zmin,zmax):
    
    coordinates.insert(0,[max(coordinates[0][0]-20,zmin),coordinates[0][1],coordinates[0][2]])
    coordinates.append([min(coordinates[-1][0]+20,zmax),coordinates[0][1],coordinates[0][2]])
    
    return coordinates

def get_local_basis(grad, *args):
    grad = grad / np.linalg.norm(grad, axis=1, keepdims=True)

    # the second basis vector must be in the sagittal plane
    sagittal = grad[:, [0, 2]]
    second = sagittal[:, ::-1] * [1, -1]

    # choose the right orientation of the basis (avoiding rotations)
    dets = np.linalg.det(np.stack([sagittal, second], -1))
    second = second * dets[:, None]
    second = second / np.linalg.norm(second, axis=1, keepdims=True)
    second = np.insert(second, 1, np.zeros_like(second[:, 0]), axis=1)

    third = np.cross(second, grad)

    return np.stack([grad, second, third], -1)

def window(img,win_min,win_max):
    #骨窗窗宽窗位
    imgmax = np.max(img)
    imgmin = np.min(img)
    if imgmax<win_max and imgmin>win_min:
        return img
    for i in range(img.shape[0]):
        img[i] = 255.0 * (img[i] - win_min) / (win_max - win_min)
        min_index = img[i] < 0
        img[i][min_index] = 0
        max_index = img[i] > 255
        img[i][max_index] = 255
    return img

import numpy as np
from scipy import ndimage

def process_layer(layer):
    processed_layer = np.zeros_like(layer)
    labels = np.unique(layer)  # 找到所有独特的label
    for label in labels:
        if label == 0:
            continue  # 跳过背景
        # 为当前label找到所有连通区域
        labeled_array, num_features = ndimage.label(layer == label)
        if num_features == 0:
            continue  # 如果没有这个label的连通区域，跳过

        leftmost_positions = []
        for feature in range(1, num_features + 1):
            # 找到当前连通区域的最左边的x坐标
            positions = np.argwhere(labeled_array == feature)
            leftmost_position = positions[:, 1].min()
            leftmost_positions.append((leftmost_position, feature))

        # 取最左边的连通区域
        if leftmost_positions:
            leftmost_positions.sort()
            _, leftmost_feature = leftmost_positions[0]
            processed_layer[labeled_array == leftmost_feature] = label

    return processed_layer

def process_3d_array(arr):
    processed_arr = np.zeros_like(arr)
    for z in range(arr.shape[2]):
        processed_arr[:,:,z] = process_layer(arr[:,:,z])
    return processed_arr


def extract_3d_volume(data, center, size=(128, 128, 64)):
    x, y, z = center
    dx, dy, dz = size
    z_min, z_max = max(0, int(z - dz // 2)), min(data.shape[2], int(z + dz // 2))
    y_min, y_max = max(0, int(y - dy // 2)), min(data.shape[1], int(y + dy // 2))
    x_min, x_max = max(0, int(x - dx // 2)), min(data.shape[0], int(x + dx // 2))

    extracted_data = data[x_min:x_max, y_min:y_max, z_min:z_max]
    # 创建一个指定大小的空白数组
    centered_volume = np.zeros(size, dtype=data.dtype)

    # 计算在centered_volume中放置extracted_data的起始索引
    start_x = (dx - (x_max - x_min)) // 2
    start_y = (dy - (y_max - y_min)) // 2
    start_z = (dz - (z_max - z_min)) // 2
    
    if start_z<0:
        centered_volume[start_x:start_x + (x_max - x_min),
                    start_y:start_y + (y_max - y_min),
                    0:size[2]] = extracted_data[:,:,0:size[2]]
    else:
        centered_volume[start_x:start_x + (x_max - x_min),
                    start_y:start_y + (y_max - y_min),
                    start_z:start_z + (z_max - z_min)] = extracted_data

    return centered_volume

def find_single_component_layers(data,label):
    
    img_data = data.copy()
    img_data[img_data!=label]=0

    # 确定中间层的索引
    mid_index = img_data.shape[2] // 2

    # 初始化记录层的变量，调整为从中间层上下各10层开始搜索
    offset = 10  # 设定的偏移量
    start_left = max(0, mid_index - offset)  # 防止索引超出下界
    start_right = min(img_data.shape[2] - 1, mid_index + offset)  # 防止索引超出上界
    z0=1
    z1=128

    # 从调整后的起点向左查找
    for i in range(start_left, -1, -1):
        layer = img_data[:, :, i]
        labeled_array, num_features = Label(layer)
        if num_features == 1:
            z0 = i
            break

    # 从调整后的起点向右查找
    for i in range(start_right, img_data.shape[2], 1):
        layer = img_data[:, :, i]
        labeled_array, num_features = Label(layer)
        if num_features == 1:
            z1 = i
            break

    return z0, z1

def find_leftmost_contour(label_binary):
    # 使用cv2.findContours找到所有轮廓
    contours, _ = cv2.findContours(label_binary.copy().astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 检查是否找到了轮廓
    if not contours:
        return None  # 如果没有找到轮廓，则返回None
    
    # 如果只有一个轮廓，直接返回这个轮廓
    if len(contours) == 1:
        return contours[0]
    
    # 如果存在多个轮廓，找到最左边的轮廓
    leftmost_x = None
    leftmost_contour = None
    for contour in contours:
        # 对于每个轮廓，计算其边界框
        x, y, w, h = cv2.boundingRect(contour)
        
        # 更新最左边的轮廓
        if leftmost_x is None or x < leftmost_x:
            leftmost_x = x
            leftmost_contour = contour
    
    return leftmost_contour

def extract_mask_volume_left(label_data,label):
    """找到最左边的连通域去包围（如果已经去除了椎弓根这一步就可以避免了）
    这一步可能会导致某些层面mask不会生成

    Args:
        label_data (_type_): _description_
        label (_type_): _description_

    Returns:
        _type_: _description_
    """
    loc = np.where(label_data == label)
    
    z0 = min(loc[2])
    z1 = max(loc[2])
    
            
    label_binary = np.zeros(label_data.shape)
    label_binary[loc] = 1
    
    other_label = np.zeros(label_data.shape)
    other_label[np.where((label_data != label) & (label_data != 0))] = 1
    
    mask_volume = np.zeros(label_data.shape)
    for slice in range(z0,z1+1):
        # 找到所有非零点的坐标
        # 找到最左边的轮廓
        try:
            contour = find_leftmost_contour(label_binary[:,:,slice])
            rect = cv2.minAreaRect(contour)
        except:
            continue
        
        # 将最小旋转矩形的四个顶点转换为整数坐标
        box = cv2.boxPoints(rect)
        rect_points = np.int0(box)
        
        # 对该最小矩形进行缩放
        # 缩放因子
        scale_factor = 1.1
        center = rect[0]
        scaled_rect_points = ((rect_points - center) * scale_factor) + center
        scaled_rect_points = np.int0(scaled_rect_points)
        
        # 创建包围椎体的最小矩形
        bbox_image = np.zeros_like(label_data[:,:,0], np.uint8)
        bbox_cv2 = cv2.cvtColor(bbox_image, cv2.COLOR_GRAY2BGR)
        cv2.fillPoly(bbox_cv2, [scaled_rect_points], [255,255,255])
        bbox_cv2 = cv2.cvtColor(bbox_cv2, cv2.COLOR_BGR2GRAY)
        
        # 应用bbox_cv2后，对label_data进行检查和处理
        # 将bbox内其他label的区域设置为0
        bbox_cv2[other_label[:,:,slice]==1]=0
                        
        mask_volume[:,:,slice] = bbox_cv2
    return mask_volume

def remove_small_connected_components(slice_img, area_threshold):
    """
    移除小的连通域。
    :param slice_img: 输入的二值化切片图像。
    :param area_threshold: 面积阈值，低于此值的连通域将被移除。
    :return: 清理后的图像。
    """
    # 寻找连通域
    contours, _ = cv2.findContours(slice_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 过滤面积小于阈值的连通域
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > area_threshold]
    # 绘制过滤后的连通域
    filtered_img = np.zeros_like(slice_img)
    cv2.drawContours(filtered_img, filtered_contours, -1, (255), thickness=cv2.FILLED)
    return filtered_img

def extract_mask_volume(label_data, label, area_threshold=20):
    """针对每个二维层面生成包围整个椎体的mask，要求是已经去除了椎弓根的
    并且预处理去除掉小的标注连通域
    这是针对3dmask生成的mask过大导致生成的椎体与实际不符；以及在extract_mask_volume_left中
    只使用最左边连通域导致某些层面的mask没有生成的错误修改的函数

    Args:
        label_data (_type_): _description_
        label (_type_): _description_
        area_threshold (int, optional): _description_. Defaults to 20.

    Returns:
        _type_: _description_
    """
    loc = np.where(label_data == label)
    z0 = min(loc[2])
    z1 = max(loc[2])
    
    other_label = np.zeros(label_data.shape)
    other_label[np.where((label_data != label) & (label_data != 0))] = 1
    
    label_binary = np.zeros(label_data.shape)
    label_binary[loc] = 1
    
    mask_volume = np.zeros(label_data.shape)
    for slice in range(z0, z1+1):
        slice_img = np.uint8(label_binary[:, :, slice] * 255)
        
        # 移除小的连通域
        cleaned_img = remove_small_connected_components(slice_img, area_threshold)
        
        # 寻找轮廓
        contours, _ = cv2.findContours(cleaned_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # 为所有找到的轮廓创建一个覆盖轮廓的最小矩形
            all_contours = np.vstack(contours[i] for i in range(len(contours)))
            rect = cv2.minAreaRect(all_contours)
            
            box = cv2.boxPoints(rect)
            rect_points = np.int0(box)
            
            # 对该最小矩形进行缩放
            scale_factor = 1.1
            center = rect[0]
            scaled_rect_points = ((rect_points - center) * scale_factor) + center
            scaled_rect_points = np.int0(scaled_rect_points)
            
            # 创建包围整体轮廓的最小矩形
            bbox_image = np.zeros_like(label_data[:, :, 0], np.uint8)
            cv2.fillPoly(bbox_image, [scaled_rect_points], 255)
            bbox_image[other_label[:,:,slice]==1]=0
            
            # 应用bbox_image
            mask_volume[:, :, slice] = bbox_image

    return mask_volume

def extract_mask_3dvolume(label_data, label):
    # 找到匹配给定标签的所有点的坐标
    loc = np.where(label_data == label)
    
    # 计算边界框的坐标
    x_min, x_max = np.min(loc[0]), np.max(loc[0])
    y_min, y_max = np.min(loc[1]), np.max(loc[1])
    z_min, z_max = np.min(loc[2]), np.max(loc[2])
    
    scale_factor = 1.1
    center = [int((x_max+x_min)/2),int((y_max+y_min)/2)]
    x_min = int((x_min - center[0]) * scale_factor + center[0])
    x_max = int((x_max - center[0]) * scale_factor + center[0])
    y_min = int((y_min - center[1]) * scale_factor + center[1])
    y_max = int((y_max - center[1]) * scale_factor + center[1])
    
    # 创建一个与label_data同形状的全零数组
    mask = np.zeros(label_data.shape, dtype=np.uint8)
    
    # 在边界框内的位置设置mask为1
    mask[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1] = 255
    mask[np.where((label_data != label) & (label_data != 0))] = 0
    
    return mask

def process_mask3d(ct_path,label_path,json_path,vertebrae_ids,output_folder,outputsize=(128,128,128)):
    
    with open(json_path, 'r') as file:
        data = json.load(file)
        
    file_size_mb = os.path.getsize(ct_path) / (1024 * 1024)
    print(file_size_mb)

    # 如果加载500mb的图像内存会不够，其实这是很傻逼的代码，把ct加载了两遍
    ct_nii = nib.load(ct_path)
    affine = ct_nii.affine
    if file_size_mb > 500:
        ct_data = ct_nii.get_fdata(dtype='float32')
    else:
        ct_data = ct_nii.get_fdata()

    # 加载标签图像，只加载一次
    label_nii = nib.load(label_path)
    label_data = label_nii.get_fdata()
    # print(np.shape(label_data)) # x,y,z
    
    
    coordinates = [[entry['X'], entry['Y'], entry['Z']] for entry in data if isinstance(entry, dict) and 'X' in entry]
    if len(coordinates)>1:
        #coordinates = extend_coordinates(coordinates,0,np.shape(label_data)[0])
        coordinates = extend_curve(np.array(coordinates),20,(0,0,0),label_data.shape)

    
    basename = os.path.basename(ct_path).replace(".nii.gz","")
    
    ct_data =  window(ct_data, -300, 800)
    shape = (128, 128) 
    #local_curve = inter.global_to_local(curve, shape=shape)
    
    # 判断文件大小是否超过阈值
    
    if len(coordinates)==1:
        print(f"Only one vertebra.")
        straight_ct = ct_data
        straight_label = label_data
    else:       
        curve = np.array(coordinates)
        inter = Interpolator(curve, step=1, get_local_basis=get_local_basis)
        straight_ct = inter.interpolate_along(ct_data, shape, order=1)
        straight_label = inter.interpolate_along(label_data, shape, order=0)

        
    straight_label = remove_spine_labels_after_split(straight_label)
    
    for i,label in enumerate(vertebrae_ids):
        output_folder_CT = os.path.join(output_folder,"CT")
        output_folder_label = os.path.join(output_folder,"label")
        output_folder_mask = os.path.join(output_folder,"mask_2d")
        
        #如果存在则跳过
        #if os.path.exists( os.path.join(output_folder_CT, basename+f"_{label}.nii.gz")):
            #print("continue")
        #    continue
        
        for folder in [output_folder_CT, output_folder_label, output_folder_mask]:
            if not os.path.exists(folder):
                os.makedirs(folder)
                print(f"Created directory: {folder}")
        

        # 这里要注意，不嫩那个把所有的json文件中的坐标都输进去，
        # 因为部分坐标是不需要的，要根据label取
        for entry in data:  # Skip the first entry which is 'direction'
            if entry['label'] == None:
                continue
            if entry['label'] == label:
                centroid = (entry['X'], entry['Y'], entry['Z'])
                if len(coordinates)>1:
                   centroid = inter.global_to_local(centroid, shape=shape)
                print(centroid)
        
        
        extracted_ct_volume = extract_3d_volume(straight_ct, centroid, size=outputsize)
        extracted_label_volume = extract_3d_volume(straight_label, centroid, size=outputsize)
        
        # find the z-axis limit
        #z0,z1= find_single_component_layers(extracted_label_volume ,label)
        #extracted_ct_volume = extracted_ct_volume[:,:,z0+4:z1-4]
        #extracted_label_volume = extracted_label_volume[:,:,z0+4:z1-4]
        
        # 去除椎弓根部分，这里就采取提取最左边连通域的方法
        #extracted_label_volume = process_3d_array(extracted_label_volume)
        
        extracted_mask_volume = extract_mask_volume(extracted_label_volume,label)
        #extracted_mask_volume = extract_mask_3dvolume(extracted_label_volume,label)
        
         # Save the extracted volume
        output_ct_path = os.path.join(output_folder_CT, basename+f"_{label}.nii.gz")
        nib.save(nib.Nifti1Image(extracted_ct_volume, affine), output_ct_path)
        
        output_label_path = os.path.join(output_folder_label, basename+f"_{label}.nii.gz")
        nib.save(nib.Nifti1Image(extracted_label_volume, affine), output_label_path)
        
        output_mask_path = os.path.join(output_folder_mask, basename+f"_{label}.nii.gz")
        nib.save(nib.Nifti1Image(extracted_mask_volume, affine), output_mask_path)
        


def parse_json(json_path):
    """
    Parses the JSON file to get patient and vertebra IDs.
    """
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data

def process_data(data_folder, data, output_folder):
    """
    Processes the specified vertebrae for each patient based on a dictionary structure.
    """
    found = False
    for category, patients in data.items():
        for patient_id, vertebrae_ids in patients.items():
            #跳过存在错误的文件
            if patient_id=="120245_series10":
               continue
            #if patient_id!="0020":
            #   continue
            
            # 直接跳过前面的文件到目标错误文件进行纠错
            #if not found:
            #    if patient_id=="YANG-CUI-YING_series0":
            #        found = True
            #    else:
            #        continue  # 跳过所有直到找到 start_file 的循环
            ct_path = os.path.join(data_folder, category, patient_id, patient_id + '.nii.gz')
            mask_path = os.path.join(data_folder, category, patient_id, patient_id + '_msk.nii.gz')
            json_path = os.path.join(data_folder, category, patient_id, patient_id + '.json')
            
            if not os.path.exists(ct_path):
                ct_path = os.path.join(data_folder, patient_id, patient_id + '.nii.gz')
                mask_path = os.path.join(data_folder, patient_id, patient_id + '_seg.nii.gz')
                json_path = os.path.join(data_folder, patient_id, patient_id + '.json')
            if not os.path.exists(ct_path):
                ct_path = find_largest_file(os.path.join(data_folder, patient_id))
            
            file_size_mb = os.path.getsize(ct_path) / (1024 * 1024)
            max_file_size_mb = 500
            #if file_size_mb > max_file_size_mb:
            #    print(patient_id)
            #    continue

            
            if os.path.exists(ct_path) and os.path.exists(mask_path) and os.path.exists(json_path):
                print(f"Processing {patient_id}: CT at {ct_path}, mask at {mask_path}, json at {json_path}")
                print(f"Vertebrae IDs: {vertebrae_ids}")
                # Here you would call your processing function with the vertebra IDs
                #try:
                process_mask3d(ct_path, mask_path, json_path, vertebrae_ids, output_folder, (256,256,64))
                #except:
                #    print("error",patient_id)
                #    continue
            else:
                print(f"Files for patient {patient_id} not found.")


def build_patient_vertebrae_map(json_path):
    """
    Builds a map of categories to patients to their vertebrae IDs from a JSON file.

    :param json_path: Path to the JSON file containing vertebra data.
    :return: A nested dictionary mapping category -> patient_id -> list of vertebrae IDs.
    """
    with open(json_path, 'r') as file:
        data = json.load(file)
    
    # Initialize the map
    category_patient_vertebrae_map = {}

    for category, patients in data.items():
        patient_vertebrae_map = {}
        for patient_vertebra_id, _ in patients.items():
            patient_id, vertebra_id = patient_vertebra_id.rsplit('_', 1)
            
            if patient_id not in patient_vertebrae_map:
                patient_vertebrae_map[patient_id] = [int(vertebra_id)]
            else:
                if vertebra_id not in patient_vertebrae_map[patient_id]:
                    patient_vertebrae_map[patient_id].append(int(vertebra_id))
        
        category_patient_vertebrae_map[category] = patient_vertebrae_map

    return category_patient_vertebrae_map

# Example usage
#data_folder = '/home/zhangqi/environments/Genant_classify/data/revised'  # Update with the actual path to your data
#json_path = '/home/zhangqi/environments/code/vertebra_data.json'  # Path to the JSON file you've uploaded
#output_folder = '/home/zhangqi/environments/data/straighten/revised'  # Update with the path where you want to save outputs

data_folder = '/mnt/g/local_dataset/preprocessed/local'  # Update with the actual path to your data
json_path = '/mnt/g/local_dataset/preprocessed/vertebra_data.json'  # Path to the JSON file you've uploaded
output_folder = '/mnt/g/local_dataset/preprocessed/straighten'  # Update with the path where you want to save outputs

#data_folder = '/mnt/g/six_local_dataset/local'  # Update with the actual path to your data
#json_path = '/mnt/g/six_local_dataset/vertebra_data.json'  # Path to the JSON file you've uploaded
#output_folder = '/mnt/g/six_local_dataset/straighten'  # Update with the path where you want to save outputs

category_patient_vertebrae_map = build_patient_vertebrae_map(json_path)

# Display the map for demonstration
for category, patients in category_patient_vertebrae_map.items():
    print(f"Category: {category}")
    for patient_id, vertebrae_ids in patients.items():
        print(f"  Patient ID: {patient_id}, Vertebrae IDs: {vertebrae_ids}")

process_data(data_folder,category_patient_vertebrae_map,output_folder)