#coding:utf-8
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import nibabel as nib
from PIL import Image
from skimage import morphology
from skimage.transform import resize
import cv2
import os
import numpy as np
from skimage import measure
import skimage
import numpy.random as npr

def get_vertbody(seg0):
    y = []
    count = []
    seg = skimage.morphology.dilation(seg0, skimage.morphology.square(2))
    label, num = measure.label(seg, connectivity=2, background=0, return_num=True)
    out = np.zeros(label.shape)
    loc_list = []
    for i in range(1, num + 1):
        loc = np.where(label == i)
        loc_list.append(loc)
        count.append(loc[0].shape[0])
        y.append(min(list(loc[1])))
    if num == 1:
        print("number=1")
        Num = 0
        countbody = np.sum(label)
    else:
        i = np.argsort(np.array(count))
        if y[i[-1]] < y[i[-2]] or count[i[-2]] < 30:

            Num = i[-1]
            countbody = count[i[-1]]
        else:
            Num = i[-2]
            countbody = count[i[-2]]

    out[loc_list[Num]] = 1
    xx = np.max(loc_list[Num][0])
    xi = np.min(loc_list[Num][0])
    yx = np.max(loc_list[Num][1])
    yi = np.min(loc_list[Num][1])
    xm = np.mean(loc_list[Num][0])
    ym = np.mean(loc_list[Num][1])
    out2 = np.zeros((60,60))
    out = out*seg0
    out2[2:3+xx-xi,2:3+yx-yi] = out[xi:xx+1,yi:yx+1]
    return out2,out,np.array([xm,ym])

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

# 采取最小旋转矩形框，使用固定scale即不进行扩增
def process_spine_data(ct_path,label_path,label_id,output_size):

        # 读取CT数据和标注数据
        #ct_data = nib.load(ct_path).get_fdata()
        #label_data = nib.load(label_path).get_fdata()
        ct_data = np.load(ct_path)
        label_data = np.load(label_path)
        binary_label = label_data.copy()
        binary_label[binary_label!=0]=255
        
        
        # 进行归一化并*255
        ct_data =  window(ct_data, -300, 800)

        label = int(label_id)
            
        
        loc = np.where(label_data == label)
        
        #if np.isnan(loc[2]):
        #    print(ct_path,label)

        try:
            center_z = int(np.mean(loc[2]))
        except:
            print("发生 ValueError 异常")
            print("loc 的值为:", loc)
            print(ct_path,label)
        _, _, center_z = np.array(np.where(label_data == label)).mean(axis=1).astype(int)

            
        # 对中间层面的椎体去除横突
        label_binary = np.zeros(label_data.shape)
        label_binary[loc] = 1
        y0 = min(loc[1])
        y1 = max(loc[1])
        z0 = min(loc[0])
        z1 = max(loc[0])

        img2d = label_binary[z0:z1 + 1, y0:y1 + 1, center_z]
            
        _, img2d_vertbody, center_point = get_vertbody(img2d)

            
        img2d_vertbody_points = np.where(img2d_vertbody==1)
        img2d_vertbody_aligned=np.zeros_like(label_data[:,:,0], np.uint8)
        # 如果将GT改为生成椎体mask，这样子就不需要纹理灰度信息了
        img2d_vertbody_aligned[img2d_vertbody_points[0]+z0,img2d_vertbody_points[1]+y0]=1
            
        # 计算椎体的中心位置
        center_y,center_x = int(np.mean(img2d_vertbody_points[0])+z0),int(np.mean(img2d_vertbody_points[1])+y0)

        # 截取224x224的矩形框在中心层面
        center_slice = ct_data[:, :, center_z].copy()
        center_label_slice = binary_label[:, :, center_z].copy()
 
        # 创建224x224的矩形框
        rect_slice = np.zeros(output_size, dtype=np.uint8)
        rect_label_slice = np.zeros(output_size, dtype=np.uint8)

        # 计算矩形框的位置
        min_y, max_y = max(0, output_size[0]//2 - center_y), min(output_size[0], output_size[0]//2 + (center_slice.shape[0] - center_y))
        min_x, max_x = max(0, output_size[0]//2 - center_x), min(output_size[0], output_size[0]//2 + (center_slice.shape[1] - center_x))

        # 将rect_slice放在中间
        rect_slice[min_y:max_y, min_x:max_x] = center_slice[max(center_y - output_size[0]//2, 0):min(center_y + output_size[0]//2, center_slice.shape[0]),
                                                                max(center_x - output_size[0]//2, 0):min(center_x +output_size[0]//2, center_slice.shape[1])]
            
        rect_label_slice[min_y:max_y, min_x:max_x] = center_label_slice[max(center_y - output_size[0]//2, 0):min(center_y + output_size[0]//2, center_slice.shape[0]),
                                                                max(center_x - output_size[0]//2, 0):min(center_x + output_size[0]//2, center_slice.shape[1])]

        # 获取椎体主体的最小旋转矩形
        contours, _ = cv2.findContours(img2d_vertbody_aligned.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rect = cv2.minAreaRect(contours[0])
            
        # 将最小旋转矩形的四个顶点转换为整数坐标
        rect_points = np.int0(cv2.boxPoints(rect))
        # 对该最小矩形进行缩放
        # 缩放因子
        scale_factor = 1.2
        center = rect[0]
        scaled_rect_points = ((rect_points - center) * scale_factor) + center
        scaled_rect_points = np.int0(scaled_rect_points)

        # 创建包围椎体的最小矩形
        bbox_image = np.zeros_like(label_data[:,:,0], np.uint8)
        bbox_cv2 = cv2.cvtColor(bbox_image, cv2.COLOR_GRAY2BGR)
            
        cv2.fillPoly(bbox_cv2, [scaled_rect_points], [255,255,255])
        bbox_cv2 = cv2.cvtColor(bbox_cv2, cv2.COLOR_BGR2GRAY)

        for other_label in range(8, 26):  # 假设label范围为1到25
            if other_label != label:
            # 找到其他label的区域
                other_label_locs = np.where(label_data[:,:,center_z] == other_label)
        
            # 检查这些区域是否在bbox内，如果在，则将这部分的masked_label设为0
                for y, x in zip(*other_label_locs):
                    if bbox_cv2[y, x] == 255:  # 如果在bbox内
                        bbox_cv2[y, x] = 0  # 将其他label区域设置为0
            


        masked_image = center_slice.copy()
        masked_image[np.where(bbox_cv2==255)[0],np.where(bbox_cv2==255)[1]] = 0
        masked_label = center_label_slice.copy()
        masked_label[np.where(bbox_cv2==255)[0],np.where(bbox_cv2==255)[1]] = 0

        masked_slice = np.zeros(output_size, dtype=np.uint8)
        masked_slice[min_y:max_y, min_x:max_x] =masked_image[max(center_y - output_size[0]//2, 0):min(center_y + output_size[0]//2, center_slice.shape[0]),
                                                                max(center_x - output_size[0]//2, 0):min(center_x +output_size[0]//2, center_slice.shape[1])]
           
        masked_label_slice = np.zeros(output_size, dtype=np.uint8)
        masked_label_slice[min_y:max_y, min_x:max_x] = masked_label[max(center_y - output_size[0]//2, 0):min(center_y + output_size[0]//2, center_slice.shape[0]),
                                                                max(center_x - output_size[0]//2, 0):min(center_x +output_size[0]//2, center_slice.shape[1])]
            
        # 保存mask区域的二值化图像
        mask_binary = np.zeros(output_size, dtype=np.uint8)
        mask_binary[min_y:max_y, min_x:max_x] = bbox_cv2[max(center_y - output_size[0]//2, 0):min(center_y + output_size[0]//2, center_slice.shape[0]),
                                                                max(center_x - output_size[0]//2, 0):min(center_x +output_size[0]//2, center_slice.shape[1])]
        
        return rect_slice,rect_label_slice,mask_binary,masked_slice,masked_label_slice


def process_spine_data_aug(ct_path,label_path,label_id,output_size):

        ct_data = np.load(ct_path)
        label_data = np.load(label_path)
        binary_label = label_data.copy()
        binary_label[binary_label!=0]=255
        
        
        # 进行归一化并*255
        ct_data =  window(ct_data, -300, 800)
        
        label = int(label_id)
   
        loc = np.where(label_data == label)
        
        try:
            center_z = int(np.mean(loc[2]))
        except:
            print("发生 ValueError 异常")
            print("loc 的值为:", loc)
            print(label_path,label)
        _, _, center_z = np.array(np.where(label_data == label)).mean(axis=1).astype(int)

        # 对中间层面的椎体去除横突
        label_binary = np.zeros(label_data.shape)
        label_binary[loc] = 1
        y0 = min(loc[1])
        y1 = max(loc[1])
        z0 = min(loc[0])
        z1 = max(loc[0])

        img2d = label_binary[z0:z1 + 1, y0:y1 + 1, center_z]
            
        _, img2d_vertbody, center_point = get_vertbody(img2d)

            
        img2d_vertbody_points = np.where(img2d_vertbody==1)
        img2d_vertbody_aligned=np.zeros_like(label_data[:,:,0], np.uint8)
        # 如果将GT改为生成椎体mask，这样子就不需要纹理灰度信息了
        img2d_vertbody_aligned[img2d_vertbody_points[0]+z0,img2d_vertbody_points[1]+y0]=1
            
            # 计算椎体的中心位置
        center_y,center_x = int(np.mean(img2d_vertbody_points[0])+z0),int(np.mean(img2d_vertbody_points[1])+y0)

        # 截取224x224的矩形框在中心层面
        center_slice = ct_data[:, :, center_z].copy()
        center_label_slice = binary_label[:, :, center_z].copy()
            #center_slice[img2d_vertbody_aligned==1]=255

        crop_height, crop_width = output_size
             # 计算椎体中心点相对于原始图像边界的最大可移动距离
        max_shift_y = min(center_y, center_slice.shape[0] - center_y, crop_height//2)/2
        max_shift_x = min(center_x, center_slice.shape[1] - center_x, crop_width//2)/2
            
            # 随机选择偏移量，保证椎体完全在裁剪图像内
        shift_y = npr.randint(-max_shift_y, max_shift_y + 1)
        shift_x = npr.randint(-max_shift_x, max_shift_x + 1)

            # 计算随机化后的裁剪起始点
        start_y = center_y + shift_y - crop_height // 2
        start_x = center_x + shift_x - crop_width // 2

            # 确定裁剪区域在原始图像内的实际位置
        actual_start_y = max(start_y, 0)
        actual_start_x = max(start_x, 0)
        actual_end_y = min(start_y + crop_height, center_slice.shape[0])
        actual_end_x = min(start_x + crop_width, center_slice.shape[1])
            
            # 创建224x224的矩形框
        rect_slice = np.zeros(output_size, dtype=np.uint8)
        rect_label_slice = np.zeros(output_size, dtype=np.uint8)

            # 将原始图像的相应区域复制到裁剪后的图像
        rect_slice[max(-start_y, 0):max(-start_y, 0)+actual_end_y-actual_start_y, 
                    max(-start_x, 0):max(-start_x, 0)+actual_end_x-actual_start_x] = \
                    center_slice[actual_start_y:actual_end_y, actual_start_x:actual_end_x]
        rect_label_slice[max(-start_y, 0):max(-start_y, 0)+actual_end_y-actual_start_y, 
                    max(-start_x, 0):max(-start_x, 0)+actual_end_x-actual_start_x] = \
                        center_label_slice[actual_start_y:actual_end_y, actual_start_x:actual_end_x]

            # 获取椎体主体的最小旋转矩形
        contours, _ = cv2.findContours(img2d_vertbody_aligned.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rect = cv2.minAreaRect(contours[0])
        contour = contours[0]
            
            # 将最小旋转矩形的四个顶点转换为整数坐标
        rect_points = np.int0(cv2.boxPoints(rect))         
            
            # 对该最小矩形进行缩放
            # 缩放因子
            # 对最小旋转矩形进行1.2-1.4之间的随机缩放
        scale_factor = npr.uniform(1.1, 1.3)
        center = rect[0]
        scaled_rect_points = ((rect_points - center) * scale_factor) + center
        scaled_rect_points = np.int0(scaled_rect_points)
            # 创建包围椎体的最小矩形
        bbox_image = np.zeros_like(label_data[:,:,0], np.uint8)
        bbox_cv2 = cv2.cvtColor(bbox_image, cv2.COLOR_GRAY2BGR)
            
        cv2.fillPoly(bbox_cv2, [scaled_rect_points], [255,255,255])
        bbox_cv2 = cv2.cvtColor(bbox_cv2, cv2.COLOR_BGR2GRAY)

            
            # 获取最小外接圆
            #(xc, yc), radius = cv2.minEnclosingCircle(contour)
            #center_circle = (int(xc), int(yc))
            #radius = int(radius*scale_factor)
            
            # 绘制最小外接圆到 bbox_cv2 上
            #cv2.circle(bbox_cv2, center_circle, radius, (255), -1)  # 用白色填充圆形
            
            # 获取最小外接矩形（非旋转）
            #x, y, w, h = cv2.boundingRect(contour)
            # 绘制最小外接矩形到 bbox_cv2 上
            #cv2.rectangle(bbox_cv2, (x, y), (x + w, y + h), (255), -1)  # 用白色填充矩形
            
            # 应用bbox_cv2后，对label_data进行检查和处理
            # 将bbox内其他label的区域设置为0
        for other_label in range(8, 26):  # 假设label范围为1到25
            if other_label != label:
                # 找到其他label的区域
                other_label_locs = np.where(label_data[:,:,center_z] == other_label)
        
                # 检查这些区域是否在bbox内，如果在，则将这部分的masked_label设为0
                for y, x in zip(*other_label_locs):
                    if bbox_cv2[y, x] == 255:  # 如果在bbox内
                        bbox_cv2[y, x] = 0  # 将其他label区域设置为0
            

            # 将椎体mask掉
        masked_image = center_slice.copy()
        masked_image[np.where(bbox_cv2==255)[0],np.where(bbox_cv2==255)[1]] = 0
        masked_label = center_label_slice.copy()
        masked_label[np.where(bbox_cv2==255)[0],np.where(bbox_cv2==255)[1]] = 0

        masked_slice = np.zeros(output_size, dtype=np.uint8)
        masked_slice[max(-start_y, 0):max(-start_y, 0)+actual_end_y-actual_start_y, 
                    max(-start_x, 0):max(-start_x, 0)+actual_end_x-actual_start_x] =\
                        masked_image[actual_start_y:actual_end_y, actual_start_x:actual_end_x]
           
        masked_label_slice = np.zeros(output_size, dtype=np.uint8)
        masked_label_slice[max(-start_y, 0):max(-start_y, 0)+actual_end_y-actual_start_y, 
                    max(-start_x, 0):max(-start_x, 0)+actual_end_x-actual_start_x] = \
                        masked_label[actual_start_y:actual_end_y, actual_start_x:actual_end_x]
            
            # 保存mask区域的二值化图像
        mask_binary = np.zeros(output_size, dtype=np.uint8)
        mask_binary[max(-start_y, 0):max(-start_y, 0)+actual_end_y-actual_start_y, 
                    max(-start_x, 0):max(-start_x, 0)+actual_end_x-actual_start_x] = \
            bbox_cv2[actual_start_y:actual_end_y, actual_start_x:actual_end_x]
                
        return rect_slice,rect_label_slice,mask_binary,masked_slice,masked_label_slice

            
