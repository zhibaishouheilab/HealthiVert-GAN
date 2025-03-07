import json
import numpy as np
import os
import nibabel as nib
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from sklearn.model_selection import ParameterGrid

def rotate_image_to_horizontal(binary_image):
    """
    Rotates the image to make the major axis of the object horizontal.
    """
    # 寻找轮廓
    binary_image = binary_image.astype(np.uint8)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 假设最大的轮廓是椎体
    contour = max(contours, key=cv2.contourArea)

    # 获取轮廓的最小外接矩形
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # 计算旋转角度
    angle = rect[2]
    if angle < -45:
        angle += 90
    if angle > 45:
        angle-=90

    # 旋转图像
    (h, w) = binary_image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(binary_image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)

    return rotated_image

def calculate_heights(segmentation_fake,segmentation_label,height_threshold):
    all_heights_fake = []
    all_heights_label = []
    pre_heights_fake = []
    pre_heights_label = []
    mid_heights_fake = []
    mid_heights_label = []
    post_heights_fake = []
    post_heights_label = []
    # 遍历z轴上的每个层
    for z in range(segmentation_label.shape[1]):
        if np.any(segmentation_label[:, z, :]) and np.any(segmentation_fake[:, z, :]):
            segmentation_label_slice = segmentation_label[:, z, :]
            segmentation_fake_slice = segmentation_fake[:, z, :]
            
            loc = np.where(segmentation_fake_slice)[1]
            y_min = int(np.min(loc))
            y_max = int(np.max(loc))
            y_range = y_max-y_min
            one_third_y = int(y_min + y_range/3)
            two_third_y = int(y_min + 2*y_range/3)
            center_height_fake = np.count_nonzero(segmentation_fake_slice[:, int(np.mean(loc))])
            all_height_fake = np.count_nonzero(segmentation_fake_slice, axis=0)
            pre_height_fake = np.count_nonzero(segmentation_fake_slice[:,:one_third_y], axis=0)
            mid_height_fake = np.count_nonzero(segmentation_fake_slice[:,one_third_y:two_third_y], axis=0)
            post_height_fake = np.count_nonzero(segmentation_fake_slice[:, two_third_y:], axis=0)
            
            loc = np.where(segmentation_label_slice)[1]
            center_height_label = np.count_nonzero(segmentation_label_slice[:, int(np.mean(loc))])
            all_height_label = np.count_nonzero(segmentation_label_slice, axis=0)
            pre_height_label = np.count_nonzero(segmentation_label_slice[:, :one_third_y], axis=0)
            mid_height_label = np.count_nonzero(segmentation_label_slice[:, one_third_y:two_third_y], axis=0)
            post_height_label = np.count_nonzero(segmentation_label_slice[:, two_third_y:], axis=0)
            

            if all_height_label.max()>all_height_fake.max():
                all_scale_ratio = all_height_label.max()/all_height_fake.max()
            else:
                all_scale_ratio = 1
            if pre_height_label.max()>pre_height_fake.max():
                pre_scale_ratio = pre_height_label.max()/pre_height_fake.max()
            else:
                pre_scale_ratio = 1
            if mid_height_label.max()>mid_height_fake.max():
                mid_scale_ratio = mid_height_label.max()/mid_height_fake.max()
            else:
                mid_scale_ratio = 1
            if post_height_label.max()>post_height_fake.max():
                post_scale_ratio = post_height_label.max()/post_height_fake.max()
            else:
                post_scale_ratio = 1
                
            all_height_fake = all_height_fake*all_scale_ratio
            center_height_fake = center_height_fake*all_scale_ratio
            pre_height_fake = pre_height_fake*pre_scale_ratio
            mid_height_fake = mid_height_fake*mid_scale_ratio
            post_height_fake = post_height_fake*post_scale_ratio
                
            all_heights_fake.extend(all_height_fake[all_height_fake > (center_height_fake*height_threshold)])  # 仅添加非零高度
            all_heights_label.extend(all_height_label[all_height_label > (center_height_label*height_threshold)])  # 仅添加非零高度
            pre_heights_fake.extend(pre_height_fake[pre_height_fake > (center_height_fake*height_threshold)])  # 仅添加非零高度
            pre_heights_label.extend(pre_height_label[pre_height_label > (center_height_label*height_threshold)])  # 仅添加非零高度
            mid_heights_fake.extend(mid_height_fake[mid_height_fake > (center_height_fake*height_threshold)])  # 仅添加非零高度
            mid_heights_label.extend(mid_height_label[mid_height_label > (center_height_label*height_threshold)])  # 仅添加非零高度
            post_heights_fake.extend(post_height_fake[post_height_fake > (center_height_fake*height_threshold)])  # 仅添加非零高度
            post_heights_label.extend(post_height_label[post_height_label > (center_height_label*height_threshold)])  # 仅添加非零高度
    
    # 将heights转换为numpy数组以便使用numpy的功能
    all_heights_fake = np.array(all_heights_fake)
    all_heights_label = np.array(all_heights_label)
    pre_heights_fake = np.array(pre_heights_fake)
    pre_heights_label = np.array(pre_heights_label)
    mid_heights_fake = np.array(mid_heights_fake)
    mid_heights_label = np.array(mid_heights_label)
    post_heights_fake = np.array(post_heights_fake)
    post_heights_label = np.array(post_heights_label)

    return all_heights_fake, all_heights_label,pre_heights_fake, pre_heights_label,mid_heights_fake, mid_heights_label,post_heights_fake, post_heights_label


def calculate_rhlv(segmentation_fake, segmentation_label, center_z, length,vertebra,height_threshold):
    """
    Calculate the Relative Height Loss Value (RHLV) between fake and label segmentations.
    """
    seg_fake_filtered = segmentation_fake[:, center_z-length:center_z+length, :]
    seg_label_filtered = segmentation_label[:, center_z-length:center_z+length, :]

    all_heights_fake, all_heights_label,pre_heights_fake, pre_heights_label,mid_heights_fake, mid_heights_label,post_heights_fake, post_heights_label\
        = calculate_heights(seg_fake_filtered, seg_label_filtered,height_threshold)
    all_height_fake = np.mean(all_heights_fake) if all_heights_fake.size > 0 else 0
    all_height_label = np.mean(all_heights_label) if all_heights_label.size > 0 else 0
    pre_height_fake = np.mean(pre_heights_fake) if pre_heights_fake.size > 0 else 0
    pre_height_label = np.mean(pre_heights_label) if pre_heights_label.size > 0 else 0
    mid_height_fake = np.mean(mid_heights_fake) if mid_heights_fake.size > 0 else 0
    mid_height_label = np.mean(mid_heights_label) if mid_heights_label.size > 0 else 0
    post_height_fake = np.mean(post_heights_fake) if post_heights_fake.size > 0 else 0
    post_height_label = np.mean(post_heights_label) if post_heights_label.size > 0 else 0
    
    all_rhlv = (all_height_fake - all_height_label) / (all_height_fake +1e-6)
    pre_rhlv = (pre_height_fake - pre_height_label) / (pre_height_fake +1e-6)
    mid_rhlv = (mid_height_fake - mid_height_label) / (mid_height_fake +1e-6)
    post_rhlv = (post_height_fake - post_height_label) / (post_height_fake +1e-6)
    min_height = min(pre_height_label,mid_height_label,post_height_label)
    max_height = max(pre_height_label,mid_height_label,post_height_label)
    relative_height_label = min_height/(max_height+1e-6)

    return all_rhlv,pre_rhlv,mid_rhlv,post_rhlv,relative_height_label

def process_datasets_to_excel(dataset_info, label_folder, fake_folder, output_file,length_divisor=5, height_threshold=0.64):
    results = []
    for dataset_type, data in dataset_info.items():
        for vertebra, label in data.items():
            label_path = os.path.join(label_folder, vertebra + '.nii.gz')
            fake_path = os.path.join(fake_folder, vertebra + '.nii.gz')

            if not os.path.exists(label_path) or not os.path.exists(fake_path):
                continue

            segmentation_label_temp = nib.load(label_path).get_fdata()
            segmentation_label = np.zeros_like(segmentation_label_temp)
            
            segmentation_fake_temp = nib.load(fake_path).get_fdata()
            segmentation_fake = np.zeros_like(segmentation_fake_temp)
            
            label_index = int(vertebra.split('_')[-1])
            segmentation_label[segmentation_label_temp == label_index] = 1
            segmentation_fake[segmentation_fake_temp == label_index] = 1
            
            loc = np.where(segmentation_label)[1]
            if loc.size == 0:
                continue  # Skip if no label index found

            min_z = np.min(loc)
            max_z = np.max(loc)
            center_z = int(np.mean(loc))
            length = (max_z - min_z) // length_divisor  # Divisor adjusted based on your setup
            

            all_rhlv, pre_rhlv, mid_rhlv, post_rhlv, relative_height_label = calculate_rhlv(
                segmentation_fake, segmentation_label, center_z, length, vertebra,height_threshold
            )
            print(pre_rhlv,mid_rhlv,post_rhlv)
            results.append({
                "Vertebra": vertebra,
                "Label": label,
                "Dataset": dataset_type,
                "All RHLV": all_rhlv,
                "Pre RHLV": pre_rhlv,
                "Mid RHLV": mid_rhlv,
                "Post RHLV": post_rhlv,
                "Relative Height Label": relative_height_label
            })

    # Create a DataFrame from results and save to Excel
    df = pd.DataFrame(results)
    df.to_excel(output_file, index=False)

def main():
    with open('vertebra_data.json', 'r') as file:
       json_data = json.load(file)
    
    label_folder ="datasets/straighten/revised/label"
    output_folder = 'output_3d/coronal'
    result_folder = 'RHLV_quantification'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    for root, dirs, files in os.walk(output_folder):
        for dir in dirs:
            if dir!='fine':
                continue
            
            exp_folder = os.path.join(root,dir)
            fake_folder = os.path.join(exp_folder,'label_fake')
            result_file = os.path.join(result_folder,dir+'.xlsx')
            process_datasets_to_excel(json_data, label_folder, fake_folder, result_file, length_divisor=5, height_threshold=0.7)

if __name__ == "__main__":
    main()