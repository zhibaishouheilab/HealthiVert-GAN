import torch
import numpy as np
import nibabel as nib
import os
from options.test_options import TestOptions
from models import create_model
import torchvision.transforms as transforms
from PIL import Image
from models.inpaint_networks import Generator
import torch.nn.functional as F
import math
from scipy.ndimage import label

def remove_small_connected_components(input_array, min_size):


    # 识别连通域
    structure = np.ones((3, 3), dtype=np.int32)  # 定义连通性结构
    labeled, ncomponents = label(input_array, structure)

    # 遍历所有连通域，如果连通域大小小于阈值，则去除
    for i in range(1, ncomponents + 1):
        if np.sum(labeled == i) < min_size:
            input_array[labeled == i] = 0

    # 如果输入是张量，则转换回张量

    return input_array

def load_model(model_path, netG_params, device):
    model = Generator(netG_params, True)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    model.to(device)
    return model

def numpy_to_pil(img_np):
    if img_np.dtype != np.uint8:
        raise ValueError("NumPy array should have uint8 data type.")
    img_pil = Image.fromarray(img_np)
    return img_pil

def process_nii_files(folder_path, model, output_folder, device):
    A_transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    mask_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    if not os.path.exists(os.path.join(output_folder, 'CT')):
        os.makedirs(os.path.join(output_folder, 'CT'))
    if not os.path.exists(os.path.join(output_folder, 'label')):
        os.makedirs(os.path.join(output_folder, 'label'))

    count = 0
    CAM_folder = '/home/zhangqi/Project/VertebralFractureGrading/heatmap/straighten_sagittal/binaryclass_1'
    for file_name in os.listdir(folder_path):
        #if file_name!="sub-verse004_20.nii.gz":
        #    continue
        if file_name.endswith('.nii.gz'):
            if os.path.exists(os.path.join(output_folder, 'CT_fake', file_name)):
                continue
            #if file_name!="sub-verse004_20.nii.gz":
            #    continue
            file_path = os.path.join(folder_path, file_name)
            label_path = file_path.replace('CT', 'label')
            ct_nii = nib.load(file_path)
            ct_data = ct_nii.get_fdata()
            label_nii = nib.load(label_path)
            label_data = label_nii.get_fdata()
            patient_id, vert_id = file_name[:-7].rsplit('_', 1)
            vert_id = int(vert_id)
            if int(vert_id)>8 and int(vert_id)-1 in label_data:
                vert_id_upper = int(vert_id)-1
            else:
                vert_id_upper=None
            if int(vert_id)<24 and int(vert_id)+1 in label_data:
                vert_id_bottom = int(vert_id)+1
            else:
                vert_id_bottom=None

            CAM_path_0 = os.path.join(CAM_folder, file_name[:-7]+'_0.nii.gz')
            CAM_path_1 = os.path.join(CAM_folder, file_name[:-7]+'_1.nii.gz')
            if not os.path.exists(CAM_path_0):
                CAM_path = CAM_path_1
            else:
                CAM_path = CAM_path_0
            CAM_data = nib.load(CAM_path).get_fdata() * 255
            
            vert_label = np.zeros_like(label_data)
            vert_label[label_data==vert_id]=1
            
            loc = np.where(vert_label)
    
            z0 = min(loc[2])
            z1 = max(loc[2])
            range_length = z1 - z0 + 1
            new_range_length = int(range_length * 4 / 5)
            new_z0 = z0 + (range_length - new_range_length) // 2
            new_z1 = new_z0 + new_range_length - 1

            output_ct_data = np.zeros_like(ct_data)
            output_seg_data = np.zeros_like(ct_data)
            center_index = (new_z0 + new_z1) // 2
            
            maxheight = 40
            
            for z in range(new_z0, new_z1 + 1):
                index_ratio = abs(z-center_index)/range_length*2
                index_ratio = torch.tensor([index_ratio])
                vert_label[:, :, z] = remove_small_connected_components(vert_label[:, :, z],50)
                coords = np.argwhere(vert_label[:, :, z])
                if coords.size==0:
                    continue
                x1, x2 = min(coords[:, 0]), max(coords[:, 0])
                width,length = vert_label[:,:,z].shape
                height = x2-x1
                if height>maxheight:
                    print(file_name,z)
                    x_mean = int(np.mean(coords[:, 0]))
                    x1 = x_mean-20
                    x2 = x1+40
                
                mask_x = (x1+x2)//2
                h2 = maxheight
                if mask_x<=h2//2:
                   min_x = 0
                   max_x = min_x + h2
                elif width-mask_x<=h2/2:
                   max_x = width
                   min_x = max_x -h2
                else:
                   min_x = mask_x-h2//2
                   max_x = min_x + h2
        
                mask_slice = np.zeros_like(vert_label[:,:,z]).astype(np.uint8)
                mask_slice[min_x:max_x+1] = 255
                ct_data_slice = np.zeros_like(mask_slice).astype(np.uint8)
                ct_data_slice[:min_x,:] = ct_data[:,:,z][(x1-min_x):x1,:]
                ct_data_slice[max_x:,:] = ct_data[:,:,z][x2:x2+(width-max_x),:]
                
                CAM_slice = np.zeros_like(mask_slice).astype(np.uint8)
                CAM_slice[:min_x,:] = CAM_data[:,:,z][(x1-min_x):x1,:]
                CAM_slice[max_x:,:] = CAM_data[:,:,z][x2:x2+(width-max_x),:]

                ct_batch = numpy_to_pil(ct_data_slice)
                ct_batch = A_transform(ct_batch)
                
                ori_ct = numpy_to_pil(ct_data[:,:,z].astype(np.uint8))
                ori_ct = A_transform(ori_ct)
                
                mask_batch = numpy_to_pil(mask_slice)
                mask_batch = mask_transform(mask_batch)
                
                CAM = numpy_to_pil(CAM_slice)
                CAM = mask_transform(CAM)

                ct_batch = ct_batch.unsqueeze(0).to(device)
                mask_batch = mask_batch.unsqueeze(0).to(device)
                CAM = CAM.unsqueeze(0).to(device)

                with torch.no_grad():
                    _, fake_B_mask_sigmoid, _, fake_B_raw, _,_,pred_h = model(ct_batch, mask_batch, 1-CAM,index_ratio)
                    #print(pred_h)
                pred_h = math.ceil(pred_h[0]*maxheight)

                fake_B_mask_raw = torch.where(fake_B_mask_sigmoid > 0.5, torch.ones_like(fake_B_mask_sigmoid), torch.zeros_like(fake_B_mask_sigmoid))
                output_seg_data[:, :, z] = fake_B_mask_raw.squeeze().cpu().numpy() * 255
                
                #single_image = fake_B_raw.cpu()
                #print(height)
                if pred_h<height:
                    pred_h = height
                height_diff = pred_h-height
                x_upper = x1-height_diff//2
                x_bottom = x_upper+pred_h
                single_image = torch.zeros_like(fake_B_raw)
                #print(x_upper,x_bottom)
                single_image[:,:,x_upper:x_bottom,:] = fake_B_raw[:,:,x_upper:x_bottom,:]
                ct_upper = torch.zeros_like(single_image)
                ct_upper[0,:,:x_upper,:] = ori_ct[:, height_diff//2:x1, :]
                ct_bottom = torch.zeros_like(single_image)
                #print(x2,x_bottom)
                ct_bottom[0,:,x_bottom:,:] = ori_ct[:, x2:x2+256-x_bottom, :]
                interpolated_image = single_image+ct_upper+ct_bottom


                fake_B = interpolated_image.cpu().numpy()
                output_ct_data[:, :, z] = fake_B

            new_ct_nii = nib.Nifti1Image(output_ct_data, ct_nii.affine)
            nib.save(new_ct_nii, os.path.join(output_folder, 'CT_fake', file_name))
            new_label_nii = nib.Nifti1Image(output_seg_data, ct_nii.affine)
            nib.save(new_label_nii, os.path.join(output_folder, 'label_fake', file_name))
            print(f"Now {file_name} has been generateed in {output_folder}")
            count+=1
            

def main():
    model_path = '/home/zhangqi/Project/pytorch-CycleGAN-and-pix2pix-master/checkpoints/0421_adaptive_sagittal/latest_net_G.pth'
    netG_params = {'input_dim': 1, 'ngf': 16}
    folder_path = '/home/zhangqi/Project/pytorch-CycleGAN-and-pix2pix-master/datasets/straighten/revised/CT'
    output_folder = '/home/zhangqi/Project/pytorch-CycleGAN-and-pix2pix-master/output_3d/sagittal/adaptive'
    if not os.path.exists(output_folder+'/CT_fake'):
        os.makedirs(output_folder+'/CT_fake')
    if not os.path.exists(output_folder+'/label_fake'):
        os.makedirs(output_folder+'/label_fake')
    device = 'cuda:0'

    model = load_model(model_path, netG_params, device)
    process_nii_files(folder_path, model, output_folder, device)

if __name__ == "__main__":
    main()
