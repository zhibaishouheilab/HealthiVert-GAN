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
    for file_name in os.listdir(folder_path):
        if file_name!="sub-verse020_14.nii.gz":
            continue
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

            z0 = min(loc[1])
            z1 = max(loc[1])

            for z in range(z0, z1 + 1):
                ct_batch = np.zeros((256, 256), dtype=np.uint8)
                mask_batch = np.zeros((256, 256), dtype=np.uint8)

                ct_batch[:, start_col:end_col] = ct_data[:, z, :]
                mask_batch[:, start_col:end_col] = mask_data[:,z, :]
                
                ct_batch = masked_ct[:, z, :].astype(np.uint8)
                mask_batch = mask_data[:, z, :].astype(np.uint8)
                ct_batch = numpy_to_pil(ct_batch)
                ct_batch = A_transform(ct_batch)

                mask_batch = numpy_to_pil(mask_batch)
                mask_batch = mask_transform(mask_batch)
                ct_batch[mask_batch==1]=0

                ct_batch = ct_batch.unsqueeze(0).to(device)
                mask_batch = mask_batch.unsqueeze(0).to(device)

                with torch.no_grad():
                    _, fake_B_mask_sigmoid, _, fake_B_raw, _ = model(ct_batch, mask_batch, torch.ones_like(ct_batch).to(device))
                    #print(fake_B_mask_sigmoid.shape)
                    # 输出就是([1, 1, 256, 64])大小，为什么不是256*256，明明输入的都是正方形的

                fake_B_mask_raw = torch.where(fake_B_mask_sigmoid > 0.5, torch.ones_like(fake_B_mask_sigmoid), torch.zeros_like(fake_B_mask_sigmoid))
                fake_B_mask = fake_B_mask_raw.squeeze().cpu().numpy()* 255
                label_data[:, z, :] = fake_B_mask
                fake_B = (mask_batch * fake_B_raw + ct_batch).squeeze().cpu().numpy()
                output_data[:, z, :] = fake_B

            new_ct_nii = nib.Nifti1Image(output_data, ct_nii.affine)
            nib.save(new_ct_nii, os.path.join(output_folder, 'CT_fake', file_name))
            new_label_nii = nib.Nifti1Image(label_data, ct_nii.affine)
            nib.save(new_label_nii, os.path.join(output_folder, 'label_fake', file_name))
            print(f"Now {file_name} has been generateed in {output_folder}")
            count+=1
            

def main():
    model_path = '/home/zhangqi/Project/pytorch-CycleGAN-and-pix2pix-master/checkpoints/0322_straighten_coronal_noCAM/300_net_G.pth'
    netG_params = {'input_dim': 1, 'ngf': 16}
    folder_path = '/home/zhangqi/Project/pytorch-CycleGAN-and-pix2pix-master/datasets/straighten/CT'
    output_folder = '/home/zhangqi/Project/pytorch-CycleGAN-and-pix2pix-master/output_3d/coronal'
    if not os.path.exists(output_folder+'/CT_fake'):
        os.makedirs(output_folder+'/CT_fake')
    if not os.path.exists(output_folder+'/label_fake'):
        os.makedirs(output_folder+'/label_fake')
    device = 'cuda:0'

    model = load_model(model_path, netG_params, device)
    process_nii_files(folder_path, model, output_folder, device)

if __name__ == "__main__":
    main()
