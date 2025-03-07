import os
import nibabel as nib
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import json
import pandas as pd
from sklearn.model_selection import ParameterGrid
import math

def calculate_iou(ori_seg, fake_seg):
    intersection = np.sum(ori_seg * fake_seg)
    union = np.sum(ori_seg + fake_seg > 0)
    if union == 0:
        return 0
    else:
        return intersection / union
    
def calculate_dice(ori_seg, fake_seg):
    # 计算两个分割之间的交集
    intersection = np.sum(ori_seg * fake_seg)
    # 计算两个分割之间的并集
    union = np.sum(ori_seg) + np.sum(fake_seg)
    # 如果并集为零，返回0，否则返回Dice系数
    if union == 0:
        return 0
    else:
        return 2.0 * intersection / union


def relative_volume_difference(ori_seg, fake_seg):
    volume_ori = np.sum(ori_seg)
    volume_fake = np.sum(fake_seg)
    if volume_ori == 0:
        return 0
    else:
        return np.abs(volume_ori - volume_fake) / volume_ori

def process_images(ori_ct_path, fake_ct_path, ori_seg_path, fake_seg_path):
    ori_ct = nib.load(ori_ct_path).get_fdata()
    fake_ct = nib.load(fake_ct_path).get_fdata()
    ori_seg_temp = nib.load(ori_seg_path).get_fdata()
    ori_seg = np.zeros_like(ori_seg_temp)
    fake_seg_temp = nib.load(fake_seg_path).get_fdata()
    fake_seg = np.zeros_like(fake_seg_temp)
    
    label = int(ori_seg_path[:-7].split('_')[-1])
    ori_seg[ori_seg_temp==label] = 1
    fake_seg[fake_seg_temp==label] = 1

    patch_psnr_list = []
    patch_ssim_list = []
    global_psnr_list = []
    global_ssim_list = []
    
    iou_value = calculate_iou(ori_seg, fake_seg)
    dice_value = calculate_dice(ori_seg, fake_seg)
    rv_diff = relative_volume_difference(ori_seg, fake_seg)
    
    loc = np.where(ori_seg)
    z0 = min(loc[2])
    z1 = max(loc[2])
    range_length = z1 - z0 + 1
    new_range_length = int(range_length * 4 / 5)
    new_z0 = z0 + (range_length - new_range_length) // 2
    new_z1 = new_z0 + new_range_length - 1
    


    for z in range(new_z0, new_z1 + 1):
        if np.sum(ori_seg[:,:,z]) > 400:
            coords = np.argwhere(ori_seg[:,:,z])
            x1, x2 = min(coords[:, 0]), max(coords[:, 0])

            crop_ori_ct = ori_ct[x1:x2+1, :, z]
            crop_fake_ct = fake_ct[x1:x2+1, :, z]

            psnr_value = compare_psnr(crop_ori_ct, crop_fake_ct, data_range=crop_ori_ct.max() - crop_ori_ct.min())
            ssim_value = compare_ssim(crop_ori_ct, crop_fake_ct, data_range=crop_ori_ct.max() - crop_ori_ct.min())

            if not np.isnan(psnr_value):
                patch_psnr_list.append(psnr_value)
            if not np.isnan(ssim_value):
                patch_ssim_list.append(ssim_value)
            
    for z in range(new_z0, new_z1 + 1):
        if np.sum(ori_seg[:,:,z]) > 400:
            psnr_value = compare_psnr(ori_ct[:,:,z], fake_ct[:,:,z], data_range=ori_ct[:,:,z].max() - ori_ct[:,:,z].min())
            ssim_value = compare_ssim(ori_ct[:,:,z], fake_ct[:,:,z], data_range=ori_ct[:,:,z].max() - ori_ct[:,:,z].min())

            if not np.isnan(psnr_value):
                global_psnr_list.append(psnr_value)
            if not np.isnan(ssim_value):
                global_ssim_list.append(ssim_value)
            
    avg_patch_psnr = np.mean(patch_psnr_list) if patch_psnr_list else 0  # 检查列表是否为空
    avg_patch_ssim = np.mean(patch_ssim_list) if patch_ssim_list else 0  # 检查列表是否为空
    avg_global_psnr = np.mean(global_psnr_list) if global_psnr_list else 0  # 检查列表是否为空
    avg_global_ssim = np.mean(global_ssim_list) if global_ssim_list else 0  # 检查列表是否为空



    return avg_global_psnr, avg_global_ssim, avg_patch_psnr, avg_patch_ssim, iou_value, rv_diff, dice_value

def average_metrics(lists):
    return np.mean(lists)

def main():
    ori_ct_folder = '/dssg/home/acct-milesun/zhangqi/Dataset/HealthiVert_straighten/CT'
    ori_seg_folder = '/dssg/home/acct-milesun/zhangqi/Dataset/HealthiVert_straighten/label'
    json_path = 'vertebra_data.json'
    save_folder = "evaluation/generation_metric"
    output_folder = '/dssg/home/acct-milesun/zhangqi/Project/HealthiVert-GAN_eval/output'
    with open(json_path, 'r') as file:
       vertebra_set = json.load(file)
    val_normal_vert = []
    for patient_vert_id in vertebra_set['val'].keys():
        if int(vertebra_set['val'][patient_vert_id]) == 0:
            val_normal_vert.append(patient_vert_id)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    for root, dirs, files in os.walk(output_folder):
        for dir in dirs:
            exp_folder = os.path.join(root,dir)
            fake_seg_folder = os.path.join(exp_folder,'label_fake')
            fake_ct_folder = os.path.join(exp_folder,'CT_fake')
            
            metrics_lists = {'global_psnr': [], 'global_ssim': [], 'patch_psnr': [], 'patch_ssim': [], 'iou': [], 'rv_diff': [], 'dice':[]}
            count=0
            for filename in os.listdir(ori_ct_folder):

                if filename.endswith(".nii.gz") and filename[:-7] in val_normal_vert:
                    ori_ct_path = os.path.join(ori_ct_folder, filename)
                    fake_ct_path = os.path.join(fake_ct_folder, filename)
                    ori_seg_path = os.path.join(ori_seg_folder, filename)
                    fake_seg_path = os.path.join(fake_seg_folder, filename)

                    global_psnr, global_ssim, patch_psnr, patch_ssim, iou, rv_diff, dice = process_images(
                        ori_ct_path, fake_ct_path, ori_seg_path, fake_seg_path)
                    if math.isnan(patch_psnr) or math.isnan(patch_ssim):
                        print("PSNR or SSIM returned NaN, skipping this set of images.")
                        continue
                    if patch_psnr==0 or patch_ssim==0:
                        print("PSNR or SSIM returned 0, skipping this set of images.")
                        continue
                    metrics_lists['global_psnr'].append(global_psnr)
                    metrics_lists['global_ssim'].append(global_ssim)
                    metrics_lists['patch_psnr'].append(patch_psnr)
                    metrics_lists['patch_ssim'].append(patch_ssim)
                    metrics_lists['iou'].append(iou)
                    metrics_lists['rv_diff'].append(rv_diff)
                    metrics_lists['dice'].append(dice)
                    count+=1

            # 计算总平均
            avg_metrics = {key: average_metrics(value) for key, value in metrics_lists.items()}
            
            with open(os.path.join(save_folder,dir+".txt"), "w") as file:
                for metric, value in avg_metrics.items():
                    file.write(f"Average {metric.upper()}: {value}\n")

if __name__ == "__main__":
    main()