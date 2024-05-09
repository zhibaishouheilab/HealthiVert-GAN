#假设读入的数据是nii格式的

import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import torch
from .mask_extract import process_spine_data, process_spine_data_aug
import json
import nibabel as nib
import random
import torchvision.transforms as transforms
from scipy.ndimage import label, find_objects

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


class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        
        # 读取json文件来选择训练集、测试集和验证集
        with open('/home/zhangqi/Project/pytorch-CycleGAN-and-pix2pix-master/data/vertebra_data.json', 'r') as file:
            vertebra_set = json.load(file)
            self.normal_vert_list = []
            self.abnormal_vert_list = []
        # 初始化存储normal和abnormal vertebrae的字典
        self.normal_vert_dict = {}
        self.abnormal_vert_dict = {}

        for patient_vert_id in vertebra_set[opt.phase].keys():
        # 分离patient id和vert id
            patient_id, vert_id = patient_vert_id.rsplit('_',1)
        
        # 判断该vertebra是normal还是abnormal
            if int(vertebra_set[opt.phase][patient_vert_id]) <= 1:
                self.normal_vert_list.append(patient_vert_id)
            # 如果是normal，添加到normal_vert_dict
                if patient_id not in self.normal_vert_dict:
                    self.normal_vert_dict[patient_id] = [vert_id]
                else:
                    self.normal_vert_dict[patient_id].append(vert_id)
            else:
                self.abnormal_vert_list.append(patient_vert_id)
            # 如果是abnormal，添加到abnormal_vert_dict
                if patient_id not in self.abnormal_vert_dict:
                    self.abnormal_vert_dict[patient_id] = [vert_id]
                else:
                    self.abnormal_vert_dict[patient_id].append(vert_id)
            if opt.vert_class=="normal":
                self.vertebra_id = np.array(self.normal_vert_list)
            elif opt.vert_class=="abnormal":
                self.vertebra_id = np.array(self.abnormal_vert_list)
            else:
                print("No vert class is set.")
                self.vertebra_id = None
    
        #self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.dir_AB = opt.dataroot
        #self.dir_mask = os.path.join(opt.dataroot,'mask',opt.phase) 
        #self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        #self.mask_paths = sorted(make_dataset(self.dir_mask, opt.max_dataset_size)) 
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        
    def numpy_to_pil(self,img_np):
        # 假设 img_np 是一个灰度图像的 NumPy 数组，值域在0到255
        if img_np.dtype != np.uint8:
            raise ValueError("NumPy array should have uint8 data type.")
        # 转换为灰度PIL图像
        img_pil = Image.fromarray(img_np)
        return img_pil
    


    # 按照金字塔概率选择一个slice，毕竟中间的slice包含的信息是最多的，因此尽量选择中间的slice
    def get_weighted_random_slice(self,z0, z1):
        # 计算新的范围，限制为原来范围的2/3
        range_length = z1 - z0 + 1
        new_range_length = int(range_length * 4 / 5)
    
    # 计算新范围的起始和结束索引
        new_z0 = z0 + (range_length - new_range_length) // 2
        new_z1 = new_z0 + new_range_length - 1
    
    # 计算中心索引
        center_index = (new_z0 + new_z1) // 2
    
    # 计算每个索引的权重
        weights = [1 - abs(i - center_index) / (new_z1 - new_z0) for i in range(new_z0, new_z1 + 1)]
    
        # 归一化权重使得总和为1
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
    
    # 根据权重随机选择一个层
        random_index = np.random.choice(range(new_z0, new_z1 + 1), p=normalized_weights)
        index_ratio = abs(random_index-center_index)/range_length*2
    
        return random_index,index_ratio
    
    def get_valid_slice(self,vert_label, z0, z1,maxheight):
        """
        尝试随机选取一个非空的slice。
        """
        max_attempts = 100  # 设定最大尝试次数以避免无限循环
        attempts = 0
        while attempts < max_attempts:
            slice_index,index_ratio = self.get_weighted_random_slice(z0, z1)
            vert_label[:, :, slice_index] = remove_small_connected_components(vert_label[:, :, slice_index],50)

            if np.sum(vert_label[:, :, slice_index])>50:  # 检查切片是否非空
                coords = np.argwhere(vert_label[:, :, slice_index])
                x1, x2 = min(coords[:, 0]), max(coords[:, 0])
                if x2-x1<maxheight:
                    return slice_index,index_ratio
            attempts += 1
        raise ValueError("Failed to find a non-empty slice after {} attempts.".format(max_attempts))



    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        CAM_folder = '/home/zhangqi/Project/VertebralFractureGrading/heatmap/straighten_sagittal/binaryclass_1'

        CAM_path_0 = os.path.join(CAM_folder, self.vertebra_id[index]+'_0.nii.gz')
        CAM_path_1 = os.path.join(CAM_folder, self.vertebra_id[index]+'_1.nii.gz')
        if not os.path.exists(CAM_path_0):
            CAM_path = CAM_path_1
        else:
            CAM_path = CAM_path_0
        CAM_data = nib.load(CAM_path).get_fdata() * 255
        #print(CAM_data.max())

        patient_id, vert_id = self.vertebra_id[index].rsplit('_', 1)
        vert_id = int(vert_id)

        normal_vert_list = self.normal_vert_dict[patient_id]


        ct_path = os.path.join(self.dir_AB,"CT",self.vertebra_id[index]+'.nii.gz')

        label_path = os.path.join(self.dir_AB,"label",self.vertebra_id[index]+'.nii.gz')
        #mask_path = os.path.join(self.dir_AB,"mask",self.vertebra_id[index]+'.nii.gz')
        ct_data = nib.load(ct_path).get_fdata()
        label_data = nib.load(label_path).get_fdata()
        vert_label = np.zeros_like(label_data)
        vert_label[label_data==vert_id]=1

        normal_vert_label = label_data.copy()
        if normal_vert_list:
            for normal_vert in normal_vert_list:
                normal_vert_label[normal_vert_label==int(normal_vert)]=255
            #normal_vert_label[mask_data==255]=0
            normal_vert_label[normal_vert_label!=255]=0
        else:
            normal_vert_label = np.zeros_like(label_data)

        loc = np.where(vert_label)
    
        z0 = min(loc[2])
        z1 = max(loc[2])
        maxheight = 40
        
        try:
            slice,slice_ratio = self.get_valid_slice(vert_label, z0, z1, maxheight)
            #vert_label[:, :, slice] = remove_small_connected_components(vert_label[:, :, slice],50)
            coords = np.argwhere(vert_label[:, :, slice])
            x1, x2 = min(coords[:, 0]), max(coords[:, 0])
        except ValueError as e:
            print(e)
        width,length = vert_label[:,:,slice].shape
        

            
        height = x2-x1

        #h2 = random.randint(height,2*height)
        #mask_x = random.randint(max(h2//2,x1),min(width-h2//2,x2))
        #min_x = mask_x-h2//2
        #max_x = min_x + h2
        mask_x = (x1+x2)//2
        h2 = maxheight
        if height>h2:
            print(slice,ct_path)
        if mask_x<=h2//2:
            min_x = 0
            max_x = min_x + h2
        elif width-mask_x<=h2/2:
            max_x = width
            min_x = max_x -h2
        else:
            min_x = mask_x-h2//2
            max_x = min_x + h2
        
        mask_slice = np.zeros_like(vert_label[:,:,slice])
        mask_slice[min_x:max_x] = 255
        
        #print(x1,x2,h2,mask_x,min_x,max_x)
        ct_data_slice = np.zeros_like(mask_slice)
        #print(256-max_x,x2+(256-max_x),x1-min_x)
        ct_data_slice[:min_x,:] = ct_data[:,:,slice][(x1-min_x):x1,:]
        ct_data_slice[max_x:,:] = ct_data[:,:,slice][x2:x2+(width-max_x),:]
        #ct_upper = ct_data[:,:,slice][:x1,:].copy()
        #ct_bottom = ct_data[:,:,slice][x2:,:].copy()

        normal_vert_label_slice = np.zeros_like(mask_slice)
        normal_vert_label_slice[:min_x,:] = normal_vert_label[:,:,slice][(x1-min_x):x1,:]
        normal_vert_label_slice[max_x:,:] = normal_vert_label[:,:,slice][x2:x2+(width-max_x),:]
        CAM_slice = np.zeros_like(mask_slice)
        CAM_slice[:min_x,:] = CAM_data[:,:,slice][(x1-min_x):x1,:]
        CAM_slice[max_x:,:] = CAM_data[:,:,slice][x2:x2+(width-max_x),:]
        
        A = ct_data[:,:,slice].astype(np.uint8)
        B = ct_data_slice.astype(np.uint8)
        A1 = vert_label[:,:,slice].copy()*255
        A1 = A1.astype(np.uint8)
        normal_vert_label = normal_vert_label_slice.astype(np.uint8)
        mask = mask_slice.astype(np.uint8)
        CAM = CAM_slice.astype(np.uint8)

        A = self.numpy_to_pil(A)
        B = self.numpy_to_pil(B)
        A1 = self.numpy_to_pil(A1)
        mask = self.numpy_to_pil(mask)
        normal_vert_label = self.numpy_to_pil(normal_vert_label)
        CAM = self.numpy_to_pil(CAM)
            
        # apply the same transform to both A and B
        A_transform =transforms.Compose([
            transforms.Grayscale(1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        mask_transform = transforms.Compose([
            transforms.ToTensor()
        ])

        A = A_transform(A)
        B = A_transform(B)
        A1 = mask_transform(A1)
        mask = mask_transform(mask)
        normal_vert_label = mask_transform(normal_vert_label)
        CAM = mask_transform(CAM)
        return {'A': A, 'A_mask': A1, 'mask':mask,'B':B,'height':height,'x1':x1,'x2':x2,'h2':h2,'slice_ratio':slice_ratio,
                'normal_vert':normal_vert_label,'CAM':CAM,'A_paths': ct_path, 'B_paths': ct_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.vertebra_id)

