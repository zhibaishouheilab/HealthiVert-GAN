import torch
import torch.nn.functional as F
from torch.autograd import Function
from model import Seresnet50_Contrastive
from utils import CustomLogger, calculate_confusion_matrix
import os
import cv2
from PIL import Image
import numpy as np
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from monai.networks.nets import SEresnet50
from pathlib import Path
import nibabel as nib
import random

class GradCam:
    def __init__(self, model, feature_layer):
        self.model = model
        self.feature_layer = feature_layer
        self.gradients = None
        self.model.eval()

        # 注册钩子
        self.register_hooks()

    def register_hooks(self):
        def forward_hook(module, input, output):
            self.features = output
            #print("Features captured:", self.features is not None)

        def backward_hook(module, grad_in, grad_out):
            #print(f'grad_in:{grad_in[0].size()}')
            self.gradients = grad_out[0]
            #print(grad_out[0])
            #print("Gradients captured:", self.gradients is not None)

        # 获取目标层
        for name, module in self.model.named_modules():
            #print(name)
            if name == self.feature_layer:
                #print(f'Find it:{name}')
                module.register_forward_hook(forward_hook)
                module.register_full_backward_hook(backward_hook)
            #else:
            #    print("No feature_layer")

    def generate_cam(self, input_image, target_class):
        output = self.model(input_image)
        if isinstance(output, tuple):
            print("output is tuple")
            output = output[0]  # 如果模型返回多个输出，则只取第一个
            #print(output)

        # 获取目标类别的得分
        score = output[:, target_class]

        # 反向传播，获取梯度
        self.model.zero_grad()
        score.backward(retain_graph=True)

        # 根据梯度和特征图计算权重
        gradients = self.gradients.data
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        features = self.features.data
        for i in range(pooled_gradients.size(0)):
            features[:, i, :, :] *= pooled_gradients[i]
        
        heatmap = torch.mean(features, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        heatmap /= torch.max(heatmap)

        # 可以将热力图进行后处理，如调整大小和叠加到原始图像上
        return heatmap.cpu().numpy()  # 根据需要转换为适合可视化的格式


class GradCamPlusPlus(GradCam):
    def generate_cam(self, input_image, target_class):
        output = self.model(input_image)
        if isinstance(output, tuple):
            output = output[0]

        # 获取目标类别的得分
        score = output[:, target_class]

        # 反向传播，获取梯度
        self.model.zero_grad()
        score.backward(retain_graph=True)

        gradients = self.gradients.data  # 这里是获取反向传播的梯度
        # 计算二阶导数和三阶导数的中间项
        gradient_power_2 = gradients**2
        gradient_power_3 = gradients**3

        # 计算alpha和权重
        global_sum = torch.sum(self.features.data, dim=[2, 3], keepdim=True)
        alpha_num = gradient_power_2
        alpha_denom = 2 * gradient_power_2 + global_sum * gradient_power_3 + 1e-7
        alpha = alpha_num / alpha_denom
        alpha = alpha.where(alpha_denom != 0, torch.zeros_like(alpha))

        positive_gradients = F.relu(score.exp() * gradients)  # 使用ReLU来确保只考虑正的影响
        weights = (alpha * positive_gradients).sum(dim=[2, 3], keepdim=True)

        # 根据权重和特征图计算Grad-CAM++
        cam = (weights * self.features.data).sum(dim=1, keepdim=True)
        cam = F.relu(cam)  # 应用ReLU
        cam = cam - torch.min(cam)
        cam = cam / torch.max(cam).data  # 归一化

        cam = cam.squeeze()  # 去除单维度
        heatmap = cam.cpu().detach().numpy()  # 转换为NumPy数组以便可视化

        return heatmap
    
def get_img_with_preprocess(img, transform):
    img_arr = img.get_fdata()
    z_center = int(img_arr.shape[2] / 2)
    slices = range(max(0, z_center - 15), min(img_arr.shape[2], z_center + 15))
    output_imgs = []
    output_tensors = []

    for slice in slices:
        output_img = img_arr[:, :, slice]
        output_img = output_img.astype(np.uint8)

        output_tensor = np.expand_dims(output_img.copy(), axis=-1)
        output_tensor = transform(output_tensor)
        output_imgs.append(output_img)
        output_tensors.append(output_tensor.unsqueeze(0))

    return output_imgs, torch.cat(output_tensors, 0)  # 返回图像列表和张量堆栈
    
def apply_heatmap_to_grayscale_and_save(heatmap, image, save_path, alpha=0.4, colormap=cv2.COLORMAP_JET):
    # 确保image是float32类型
    img_gray = image.astype(np.float32)
    
    # 将灰度图像转换为三通道的彩色图像
    img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    
    # 将热力图调整为原图大小
    heatmap = cv2.resize(heatmap, (img_color.shape[1], img_color.shape[0]))
    
    # 确保heatmap也是float32类型
    heatmap = np.uint8(255 * heatmap.astype(np.float32))

    heatmap = cv2.applyColorMap(heatmap, colormap)

    # 叠加热力图与原图
    superimposed_img = heatmap * alpha + img_color  # 调整因alpha叠加需要
    superimposed_img = superimposed_img / np.max(superimposed_img) * 255
    superimposed_img = np.uint8(superimposed_img)
    
    # 保存图像
    cv2.imwrite(save_path, superimposed_img)
    print(f"Image saved to {save_path}")


def process_and_save_nii(dataroot, output_folder, grad_cam, target_class=1):
    folder_path = Path(os.path.join(dataroot, 'test'))
    if not os.path.exists(os.path.join(dataroot, 'test')):
        folder_path = Path(dataroot)
    nii_files = list(folder_path.rglob('*.nii.gz'))

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])

    for img_nii in nii_files:
        filename = img_nii.stem.split('.')[0]
        img_nii = nib.load(str(img_nii))
        img_arr = img_nii.get_fdata()
        image_raw_list, input_tensor = get_img_with_preprocess(img_nii, transform)
        input_tensor = input_tensor.cuda(0, non_blocking=True).float()

        heatmap_3d = np.zeros(img_nii.get_fdata().shape)  # 初始化三维热力图数组

        # 处理中间30层
        z_center = int(img_nii.shape[2] / 2)
        start_slice = max(0, z_center - 15)
        end_slice = min(img_nii.shape[2], z_center + 15)
        for i, slice_idx in enumerate(range(start_slice, end_slice)):
            input_slice = input_tensor[i:i+1]  # 获取当前层的输入张量
            heatmap = grad_cam.generate_cam(input_slice, target_class)  # 生成当前层的热力图
            heatmap_resized = cv2.resize(heatmap, (img_nii.shape[0], img_nii.shape[1]))  # 调整热力图大小
            heatmap_3d[:, :, slice_idx] = heatmap_resized

        # 保存为NIfTI格式
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        save_path = os.path.join(output_folder, filename + '.nii.gz')
        new_img_nii = nib.Nifti1Image(heatmap_3d, img_nii.affine, img_nii.header)
        nib.save(new_img_nii, save_path)
        print(f"NIfTI saved to {save_path}")
        # 保存中间层作为演示效果图
        imgsave_path = os.path.join(output_folder, filename + '.png')

        apply_heatmap_to_grayscale_and_save(heatmap_3d[:,:,z_center],img_arr[:,:,z_center],imgsave_path)
    


# 使用示例
torch.cuda.set_device(0)
ckpt_path = '/home/zhangqi/Project/VertebralFractureGrading/ckpt/binary_straighten_0406_sagittal'
checkpoint = torch.load(os.path.join(ckpt_path,"best_ckpt_41136_0.9758208075449455.tar"),
                                    map_location=torch.device('cuda', 0))
model = SEresnet50(spatial_dims=2, in_channels=1, num_classes=2)  # 根据实际情况初始化你的模型
model = torch.nn.DataParallel(model).cuda()

model.load_state_dict(checkpoint['state_dict'])
target_layers = [model.module.layer4[-1]]

#grad_cam = GradCam(model=model, feature_layer="module.layer4.2.conv1.conv")  # 确保feature_layer与模型中的层名称相匹配
#grad_cam = GradCAM(model=model, target_layers=target_layers)
grad_cam = GradCamPlusPlus(model=model, feature_layer="module.layer4.2.conv1.conv")

#dataroot = '/home/zhangqi/Project/pytorch-CycleGAN-and-pix2pix-master/datasets/straighten/revised/binaryclass'
dataroot = '/home/zhangqi/Project/pytorch-CycleGAN-and-pix2pix-master/datasets/local/straighten/CT'
target_class=1

#output_folder = f'/home/zhangqi/Project/VertebralFractureGrading/heatmap/straighten_sagittal/binaryclass_{target_class}'
output_folder = f'/home/zhangqi/Project/VertebralFractureGrading/heatmap/local_sagittal_0508/binaryclass_{target_class}'
process_and_save_nii(dataroot, output_folder, grad_cam, target_class=target_class)
