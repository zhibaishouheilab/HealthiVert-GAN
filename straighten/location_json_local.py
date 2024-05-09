#对于本地数据集，查找对应的mask中的label，并对于每个label找到其重心位置作为中心点
#保存在json文件中；这对于后续的straighten有作用


import json
import nibabel as nib
import numpy as np
import os

def load_nifti_data(file_path):
    nii = nib.load(file_path)
    return nii.get_fdata().astype(np.uint8)

def calculate_center_of_mass(data, label):
    center = np.mean(np.where(data == label), axis=1)
    return center # Reverse the order to match X, Y, Z

def process_directory(root_dir):

    for base_filename in os.listdir(root_dir):
        
        #if base_filename!='0020':
        #    continue
        print(base_filename)
        original_nifti_path = os.path.join(root_dir,base_filename, base_filename+'_seg.nii.gz')
        if not os.path.exists(original_nifti_path):
            original_nifti_path = os.path.join(root_dir,base_filename, base_filename+'_msk.nii.gz')
        json_path = os.path.join(root_dir,base_filename, f'{base_filename}.json')
        #if os.path.exists(json_path):
        #    continue
        
        data_orig = nib.load(original_nifti_path).get_fdata().astype(np.uint8)
        labels_orig = np.unique(data_orig)
        labels_orig = labels_orig[labels_orig !=0]
        
        if not os.path.exists(os.path.dirname(json_path)):
            os.makedirs(os.path.dirname(json_path))
            
        json_data = []
        for label in labels_orig:
            if label==0:
                continue
            if np.sum(data_orig==label)<8000 and label==max(labels_orig):
                continue
            if np.sum(data_orig==label)<6000 and label==min(labels_orig):
                continue
            center = calculate_center_of_mass(data_orig, label)
            json_data.append({"label": int(label), "X": center[0], "Y": center[1], "Z": center[2]})

        json_data.sort(key=lambda x: x.get("label", 0))
        
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=4)

root_dir =  '/mnt/g/local_dataset/preprocessed/local'
#root_dir =  '/mnt/g/six_local_dataset/local'
process_directory(root_dir)