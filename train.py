"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from options.test_options import TestOptions
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import os 
import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
import torch.nn.functional as F
import math

def dice_score(pred, target, smooth=1e-5):
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    intersection = (pred_flat * target_flat).sum()
    return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

def iou_score(pred, target, smooth=1e-5):
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    return (intersection + smooth) / (union + smooth)

def evaluate_model(model, test_loader, device,checkpoint_path, iteration):
    model.eval()  # 将模型设置为评估模式
    with torch.no_grad():  # 关闭梯度计算
        ssim_scores, psnr_scores, dice_scores, iou_scores = [], [], [], []
        Diff_hs = []
        for batch in test_loader:
            model.set_input(batch)
            
            ground_truths, labels, normal_vert_labels, masks,CAMs,heights,x1,x2,slice_ratio = \
                model.real_B,model.real_B_mask,model.normal_vert,model.mask,model.CAM,model.height,\
                    model.x1,model.x2,model.slice_ratio
            maxheight = model.maxheight
            ct_upper_list = []
            ct_bottom_list = []
            for i in range(ground_truths.shape[0]):
                ct_upper = ground_truths[i, :, :x1[i], :]
                ct_bottom = ground_truths[i, :, x2[i]:, :]
                ct_upper_list.append(ct_upper.unsqueeze(0))  # 添加批次维度以便合并
                ct_bottom_list.append(ct_bottom.unsqueeze(0))
                
            # 模型推理
            CAM_temp = 1-CAMs
            inputs = model.real_A
            outputs = model.netG(inputs,masks,CAM_temp,slice_ratio)  # 根据你的模型调整
            coarse_seg_sigmoid,fine_seg_sigmoid, stage1, stage2, offset_flow,pred1_h,pred2_h = outputs  # 根据你的输出调整
            pred1_h = pred1_h.T*maxheight
            pred2_h = pred2_h.T*maxheight

            coarse_seg_binary = torch.where(coarse_seg_sigmoid>0.5,torch.ones_like(coarse_seg_sigmoid),torch.zeros_like(coarse_seg_sigmoid))
            fine_seg_binary = torch.where(fine_seg_sigmoid>0.5,torch.ones_like(fine_seg_sigmoid),torch.zeros_like(fine_seg_sigmoid))
            
            fake_B_raw_list = []
            for i in range(stage2.size(0)):
                height = math.ceil(pred2_h[0][i].item())  # 获取当前图片的目标高度
                if height<heights[i]:
                    height = heights[i]
                height_diff = height-heights[i]
                x_upper = x1[i]-height_diff//2
                x_bottom = x_upper+height
                single_image = torch.zeros_like(stage2[i:i+1])
                single_image[0,:,x_upper:x_bottom,:] = stage2[i:i+1,:,x_upper:x_bottom,:]
                ct_upper = torch.zeros_like(single_image)
                ct_upper[0,:,:x_upper,:] = ground_truths[i, :, height_diff//2:x1[i], :]
                ct_bottom = torch.zeros_like(single_image)
                ct_bottom[0,:,x_bottom:,:] = ground_truths[i, :, x2[i]:x2[i]+256-x_bottom, :]
                interpolated_image = single_image+ct_upper+ct_bottom
                fake_B_raw_list.append(interpolated_image)


            inpainted_result = torch.cat(fake_B_raw_list, dim=0)
            
            # 计算评估指标
            # 注意：你需要将Tensor转换为适合评估函数的numpy数组，且可能需要处理多个样本的batch
            for i in range(inputs.size(0)):  # 遍历batch中的每个样本
                # 这里添加从Tensor到numpy的转换，以及任何必要的预处理步骤
                # 假设ground_truth, label, normal_vert_label已经是正确格式的numpy数组
                ground_truth = ground_truths[i].cpu().numpy()
                label = labels[i].cpu().numpy()
                normal_vert_label = normal_vert_labels[i].cpu().numpy()
                height = heights[i].cpu()
                pred_h = pred2_h[0][i].cpu()
                
                # 假设函数inpainted_result_to_numpy等可以正确转换模型输出
                inpainted_result_np = inpainted_result[i].cpu().numpy()
                coarse_seg_binary_np = coarse_seg_binary[i].cpu().numpy()
                fine_seg_binary_np = fine_seg_binary[i].cpu().numpy()
                mask = masks[i].cpu().numpy()
                
                
                # 示例：计算SSIM
                # 注意，直接把整张图像输入计算SSIM会导致背景区域影响很大
                # 可以结合二值化mask只对前景区域计算SSIM指数
                ssim_score = ssim((ground_truth*mask).squeeze(), (inpainted_result_np*mask).squeeze(), data_range=inpainted_result_np.max() - inpainted_result_np.min(), multichannel=True)
                ssim_scores.append(ssim_score)
                
                image_psnr = psnr((ground_truth*mask).squeeze(), (inpainted_result_np*mask).squeeze(), data_range=inpainted_result_np.max() - ground_truth.min())
                psnr_scores.append(image_psnr)
                
                dice_value_coarse = dice_score(torch.tensor(coarse_seg_binary_np).float(), torch.tensor(normal_vert_label).float())
                dice_scores.append(dice_value_coarse)
                
                iou_value_fine = iou_score(torch.tensor(fine_seg_binary_np).float(), torch.tensor(label).float())
                iou_scores.append(iou_value_fine)
                # 示例：计算PSNR, Dice, IoU等
                Diff_h = (abs(pred_h-height)/height)*100
                #print(Diff_h.cpu())
                Diff_hs.append(Diff_h)
                

        # 在这里计算整个测试集上的评估指标平均值
        #print(ssim_scores)
        avg_ssim = np.mean(ssim_scores)
        avg_psnr = np.mean(psnr_scores)
        avg_dice = np.mean(dice_scores)
        avg_iou = np.mean(iou_scores)

        avg_diffh = np.mean(Diff_hs)
        # 计算其他指标的平均值...
        
    model.train()  # 恢复模型到训练模式
    viz_images = torch.stack([inputs, inpainted_result,ground_truths,
                                              coarse_seg_binary,normal_vert_labels,fine_seg_binary,labels,CAMs], dim=1)
    viz_images = viz_images.view(-1, *list(inputs.size())[1:])
    imgsave_pth =os.path.join(checkpoint_path,"eval_imgs")
    if not os.path.exists(imgsave_pth):
        os.makedirs(imgsave_pth)
    vutils.save_image(viz_images,
                                  '%s/nepoch_%03d_eval.png' % (imgsave_pth, iteration),
                                  nrow=3 * 4,
                                  normalize=True)
    return avg_ssim, avg_psnr, avg_dice, avg_iou, avg_diffh  # 返回计算出的平均指标

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    logdir=os.path.join(opt.checkpoints_dir, opt.name,'checkpoints')
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    writer = SummaryWriter(logdir=logdir)
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)
    
    # test setting
    opt_test = TestOptions().parse()  # get test options
    opt_test.batch_size = 5    # test code only supports batch_size = 1
    opt_test.serial_batches = True
    opt_test.phase = "test"
    dataset_test = create_dataset(opt_test)  # create a dataset given opt.dataset_mode and other options

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)
            
        # 经过15个epoch评估一次
        if epoch % 15==0:
            avg_ssim, avg_psnr, avg_dice, avg_iou,avg_diffh = evaluate_model(model, dataset_test, "cuda:0",os.path.join(opt.checkpoints_dir, opt.name),epoch)
             # 记录评估指标
            writer.add_scalar('Eval/SSIM', avg_ssim, epoch)
            writer.add_scalar('Eval/PSNR', avg_psnr, epoch)
            writer.add_scalar('Eval/Dice', avg_dice, epoch)
            writer.add_scalar('Eval/IoU', avg_iou, epoch)
            writer.add_scalar('Eval/DiffH', avg_diffh, epoch)
            print(f'epoch[{epoch}/{opt.n_epochs + opt.n_epochs_decay + 1}], SSIM: {avg_ssim}, PSNR: {avg_psnr}, Dice: {avg_dice}, IoU: {avg_iou}, Diffh: {avg_diffh}')


        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
