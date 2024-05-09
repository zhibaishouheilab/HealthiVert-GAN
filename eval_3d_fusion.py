import torch
import numpy as np
import nibabel as nib
import os
from options.test_options import TestOptions
from models import create_model
import torchvision.transforms as transforms
from PIL import Image
from models.inpaint_networks import Generator

def load_model(model_path, netG_params, device):
    model = Generator(netG_params, True)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    model.to(device)
    return model

def normalize_tensor(tensor):
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    # 防止除以0的情况
    if tensor_max - tensor_min > 0:
        normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
    else:
        # 如果最大值等于最小值，可能需要根据实际情况处理，这里我们选择返回原tensor
        normalized_tensor = tensor
    return normalized_tensor


def numpy_to_pil(img_np):
    if img_np.dtype != np.uint8:
        raise ValueError("NumPy array should have uint8 data type.")
    img_pil = Image.fromarray(img_np)
    return img_pil

def process_nii_files(folder_path, model_sigittal, model_coronal, output_folder, device):
    A_transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    mask_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    if not os.path.exists(os.path.join(output_folder, 'CT_fake')):
        os.makedirs(os.path.join(output_folder, 'CT_fake'))
    if not os.path.exists(os.path.join(output_folder, 'label_fake')):
        os.makedirs(os.path.join(output_folder, 'label_fake'))

    count = 0
    for file_name in os.listdir(folder_path):
        #if count>10:
        #    break
        if file_name.endswith('.nii.gz'):
            file_path = os.path.join(folder_path, file_name)
            mask_path = file_path.replace('CT', 'mask')
            ct_nii = nib.load(file_path)
            ct_data = ct_nii.get_fdata()
            mask_nii = nib.load(mask_path)
            mask_data = mask_nii.get_fdata()
            
            masked_ct = ct_data.copy()
            
            output_data = np.zeros_like(ct_data)
            label_data = np.zeros_like(ct_data)

            start_col = (256 - 64) // 2
            end_col = start_col + 64

            loc = np.where(mask_data)

            y0 = min(loc[1])
            y1 = max(loc[1])
            z0 = min(loc[2])
            z1 = max(loc[2])
            
            # 如果想要获得对比度正确的CT图像，就需要在每一层都进行正则化,这样子的问题就是速度很慢
            # 对于label的获取是无所谓的

            for y in range(0, ct_data.shape[1]):
                ct_batch = np.zeros((256, 256), dtype=np.uint8)
                mask_batch = np.zeros((256, 256), dtype=np.uint8)

                ct_batch[:, start_col:end_col] = ct_data[:, y, :]
                mask_batch[:, start_col:end_col] = mask_data[:,y, :]
                
                ct_batch = masked_ct[:, y, :].astype(np.uint8)
                mask_batch = mask_data[:, y, :].astype(np.uint8)
                ct_batch = numpy_to_pil(ct_batch)
                ct_batch = A_transform(ct_batch)

                mask_batch = numpy_to_pil(mask_batch)
                mask_batch = mask_transform(mask_batch)
                ct_batch[mask_batch==1]=0
                
                # 注意都是相加，而不是直接等于；不然会影响到另一个维度
                # 其实最保险的操作就是分开计算两个output最后取个平均
                output_data[:,y, :] += ct_batch.squeeze().numpy()

                ct_batch = ct_batch.unsqueeze(0).to(device)
                mask_batch = mask_batch.unsqueeze(0).to(device)

                if y in range(y0, y1 + 1):
                    with torch.no_grad():
                        _, fake_B_mask_sigmoid, _, fake_B_raw, _ = model_coronal(ct_batch, mask_batch, torch.ones_like(ct_batch).to(device))
                    #print(fake_B_mask_sigmoid.shape)
                    # 输出就是([1, 1, 256, 64])大小，为什么不是256*256，明明输入的都是正方形的

                #fake_B_mask_raw = torch.where(fake_B_mask_sigmoid > 0.5, torch.ones_like(fake_B_mask_sigmoid), torch.zeros_like(fake_B_mask_sigmoid))
                    fake_B_mask = fake_B_mask_sigmoid.squeeze().cpu().numpy()
                    label_data[:,y, :] += fake_B_mask
                # 归一化需不需要 ct_batch？
                    #fake_B = normalize_tensor(mask_batch * fake_B_raw + ct_batch)
                    fake_B = mask_batch * fake_B_raw
                    fake_B = fake_B.squeeze().cpu().numpy()
                    output_data[:,y, :] += fake_B
                
            for z in range(0, ct_data.shape[2]):
                ct_batch = masked_ct[:, :, z].astype(np.uint8)
                mask_batch = mask_data[:, :, z].astype(np.uint8)
                ct_batch = numpy_to_pil(ct_batch)
                ct_batch = A_transform(ct_batch)

                mask_batch = numpy_to_pil(mask_batch)
                mask_batch = mask_transform(mask_batch)
                ct_batch[mask_batch==1]=0
                
                output_data[:, :, z] += ct_batch.squeeze().numpy()

                ct_batch = ct_batch.unsqueeze(0).to(device)
                mask_batch = mask_batch.unsqueeze(0).to(device)

                if z in range(z0, z1 + 1):
                    with torch.no_grad():
                        _, fake_B_mask_sigmoid, _, fake_B_raw, _ = model_sigittal(ct_batch, mask_batch, torch.ones_like(ct_batch).to(device))

                #fake_B_mask_raw = torch.where(fake_B_mask_sigmoid > 0.5, torch.ones_like(fake_B_mask_sigmoid), torch.zeros_like(fake_B_mask_sigmoid))
                    label_data[:, :, z] += fake_B_mask_sigmoid.squeeze().cpu().numpy()
                    #fake_B = normalize_tensor(mask_batch * fake_B_raw + ct_batch)
                    fake_B = mask_batch * fake_B_raw
                    fake_B = fake_B.squeeze().cpu().numpy()
                    output_data[:, :, z] += fake_B
            
            label_data_binary = np.where(label_data > 1, np.ones_like(label_data), np.zeros_like(label_data))
            output_data_fusion = output_data/2

            new_ct_nii = nib.Nifti1Image(output_data_fusion, ct_nii.affine)
            nib.save(new_ct_nii, os.path.join(output_folder, 'CT_fake', file_name))
            new_label_nii = nib.Nifti1Image(label_data_binary, ct_nii.affine)
            nib.save(new_label_nii, os.path.join(output_folder, 'label_fake', file_name))
            print(f"Now {file_name} has been generateed in {output_folder}")
            count+=1
            

def main():
    model_path_sagittal = '/home/zhangqi/Project/pytorch-CycleGAN-and-pix2pix-master/checkpoints/0321_straighten_randomslice_noCAM/240_net_G.pth'
    model_path_coronal = '/home/zhangqi/Project/pytorch-CycleGAN-and-pix2pix-master/checkpoints/0322_straighten_coronal_noCAM/300_net_G.pth'
    netG_params = {'input_dim': 1, 'ngf': 16}
    folder_path = '/home/zhangqi/Project/pytorch-CycleGAN-and-pix2pix-master/datasets/straighten/revised/CT'
    output_folder = '/home/zhangqi/Project/pytorch-CycleGAN-and-pix2pix-master/output_3d/fusion/revised'
    if not os.path.exists(output_folder+'/CT_fake'):
        os.makedirs(output_folder+'/CT_fake')
    if not os.path.exists(output_folder+'/label_fake'):
        os.makedirs(output_folder+'/label_fake')
    device = 'cuda:0'

    model_sagittal = load_model(model_path_sagittal, netG_params, device)
    model_conoral = load_model(model_path_coronal, netG_params, device)
    process_nii_files(folder_path, model_sagittal, model_conoral, output_folder, device)

if __name__ == "__main__":
    main()
