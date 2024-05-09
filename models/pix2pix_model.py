import torch
from .base_model import BaseModel
from . import networks
import torch.nn as nn
from .UnetG_CT_mask import define_G
from .inpaint_networks import Generator
from .edge_operator import edge_loss,Sobel
import torch.nn.functional as F
import random
import math


def diceCoeff(pred, gt, eps=1e-5, activation='sigmoid'):
    r""" computational formula：
        dice = (2 * tp) / (2 * tp + fp + fn)
    """
 
    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = nn.Softmax2d()
    else:
        raise NotImplementedError("Activation implemented for sigmoid and softmax2d 激活函数的操作")
 
    pred = activation_fn(pred)
    
    #print(torch.min(gt),torch.max(gt))
    N = gt.shape[0]
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)
 
    tp = torch.sum(gt_flat * pred_flat, dim=1)
    fp = torch.sum(pred_flat, dim=1)
    fn = torch.sum(gt_flat, dim=1) 
    loss = (2 * tp + eps) / (fp + fn + eps)
    #print(tp,fp,fn,loss)
    return loss.sum() / N

class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=200.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_maskL1','G_Dice','coarse_Dice','edge',\
            'D_real_1', 'D_fake_1','D_real_2', 'D_fake_2', 'D_real_3','D_fake_3','h']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        #self.visual_names = ['real_A','real_A_mask', 'fake_B','fake_B_mask', 'real_B','real_B_mask','mask','fake_edges','real_edges']
        self.visual_names = ['real_A', 'fake_B','fake_B_mask_raw','normal_vert','coarse_seg_binary',\
            'fake_B_coarse', 'real_B','mask','fake_B_raw','real_B_mask','CAM','real_edges','fake_B_local']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D_1','D_2', 'D_3']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        
        # 这里使用了label作为第二个通道,mask作为第三个通道，所以输入通道要+2
        #self.netG = networks.define_G(opt.input_nc+2, opt.output_nc, opt.ngf, opt.netG, opt.norm,
        #                              not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        #self.netG = define_G(opt.input_nc+2, opt.output_nc, opt.ngf, opt.netG, opt.norm,
        #                              not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        
        # using inpainting network
        netG_params = {'input_dim': 1, 'ngf': 16}
        self.netG = Generator(netG_params,True)
        self.netG.cuda()
        
        #定义一个边缘提取器
        self.sobel_edge = Sobel(requires_grad=False).to(self.device)
        self.edge_loss = F.mse_loss
        
        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            #self.netD_1 = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
            #                              opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_1 = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_2 = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_3 = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            # 感觉生成器的性能不如判别器。加大学习率试一下
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_1 = torch.optim.Adam(self.netD_1.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_2 = torch.optim.Adam(self.netD_2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_3 = torch.optim.Adam(self.netD_3.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D_1)
            self.optimizers.append(self.optimizer_D_2)
            self.optimizers.append(self.optimizer_D_3)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        #self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.real_B_mask = input['A_mask'].to(self.device)
        self.real_A= input['A' if AtoB else 'B'].to(self.device)
        self.CAM = input['CAM'].to(self.device)
        self.normal_vert = input['normal_vert'].to(self.device)
        self.height = input['height'].to(self.device)
        self.mask = input['mask'].to(self.device)
        #self.h2 = input['h2']
        self.slice_ratio = input['slice_ratio'].to(self.device)

        self.x1 = input['x1'].to(self.device)
        self.x2 = input['x2'].to(self.device)
        #ct_upper_list = []
        #ct_bottom_list = []
        #for i in range(self.real_B.shape[0]):
        #    ct_upper = torch.zeros_like(self.real_B[i, :, :, :])
        #    ct_bottom = torch.zeros_like(self.real_B[i, :, :, :])
        #    ct_upper[:, :self.x1[i], :] = self.real_B[i, :, :self.x1[i], :]
        #    ct_bottom[:, :self.x2[i], :] = self.real_B[i, :, self.x2[i]:, :]
        #    ct_upper_list.append(ct_upper.unsqueeze(0))  # 添加批次维度以便合并
        #    ct_bottom_list.append(ct_bottom.unsqueeze(0))

        #self.ct_upper_list = ct_upper_list
        #self.ct_bottom_list = ct_bottom_list
        self.maxheight = input['h2'].to(self.device)

        

        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        #print(self.image_paths)
       # self.A_1 = input['A_1'].to(self.device)
       # self.A_2 = input['A_2'].to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        #self.fake_B,self.fake_B_mask_sigmoid = self.netG(torch.cat((self.real_A, self.real_A_mask,self.mask), 1))  # G(A)
        
        # 还可以做一个mask的分割任务，用来监督mask区域的生成效果

        CAM_temp = 1-self.CAM
        #print(self.CAM.max())
        self.coarse_seg_sigmoid,self.fake_B_mask_sigmoid,self.x_stage1,self.fake_B_raw,\
            self.offset_flow,self.pred1_h,self.pred2_h = self.netG(self.real_A,self.mask,CAM_temp,self.slice_ratio)  # G(A)
        
        self.pred1_h = self.pred1_h.T*self.maxheight
        self.pred2_h = self.pred2_h.T*self.maxheight
        

        #print(self.pred1_h,self.pred2_h,self.height)

        #self.fake_B_mask_sigmoid = nn.Sigmoid()(self.fake_B_mask_sigmoid)

        # sigmoid函数返回值范围为0-1
        #print(torch.min(self.fake_B_mask_sigmoid))
        self.fake_B_mask_raw = torch.where(self.fake_B_mask_sigmoid>0.5,torch.ones_like(self.fake_B_mask_sigmoid),torch.zeros_like(self.fake_B_mask_sigmoid))
        self.coarse_seg_binary = torch.where(self.coarse_seg_sigmoid>0.5,torch.ones_like(self.coarse_seg_sigmoid),torch.zeros_like(self.coarse_seg_sigmoid))

        #self.x_stage1 = F.interpolate(self.x_stage1, size=(self.pred1_h.item(),self.x_stage1.shape[2]), mode='bilinear', align_corners=False)

        #self.fake_B_raw = F.interpolate(self.fake_B_raw, size=(self.pred2_h.item(),self.fake_B_raw.shape[2]), mode='bilinear', align_corners=False)
        fake_B_raw_list = []
        for i in range(self.fake_B_raw.size(0)):
            height = math.ceil(self.pred2_h[0][i].item())
            if height<self.height[i]:
                height = self.height[i]
            height_diff = height-self.height[i]
            #height = math.ceil(self.pred2_h[0][i].item())  # 获取当前图片的目标高度
            #if height<self.height[i]:
            #    height = self.height[i]
            x_upper = self.x1[i]-height_diff//2
            x_bottom = x_upper+height
            single_image = torch.zeros_like(self.fake_B_raw[i:i+1])
            single_image[0,:,x_upper:x_bottom,:] = self.fake_B_raw[i:i+1,:,x_upper:x_bottom,:]
            ct_upper = torch.zeros_like(single_image)
            ct_upper[0,:,:x_upper,:] = self.real_B[i, :, height_diff//2:self.x1[i], :]
            ct_bottom = torch.zeros_like(single_image)
            ct_bottom[0,:,x_bottom:,:] = self.real_B[i, :, self.x2[i]:self.x2[i]+256-x_bottom, :]
            
            interpolated_image = single_image+ct_upper+ct_bottom
            #interpolated_image = F.interpolate(single_image, size=(height, single_image.shape[3]), mode='bilinear', align_corners=False)
            fake_B_raw_list.append(interpolated_image)

        #self.fake_B_raw = torch.cat(interpolated_images, dim=0)
        
        x_stage1_list = []
        for i in range(self.x_stage1.size(0)):
            height = math.ceil(self.pred1_h[0][i].item())  # 获取当前图片的目标高度
            if height<self.height[i]:
                height = self.height[i]
            height_diff = height-self.height[i]
            
            x_upper = self.x1[i]-height_diff//2
            x_bottom = x_upper+height
            single_image = torch.zeros_like(self.x_stage1[i:i+1])
            single_image[0,:,x_upper:x_bottom,:] = self.x_stage1[i:i+1,:,x_upper:x_bottom,:]
            ct_upper = torch.zeros_like(single_image)
            ct_upper[0,:,:x_upper,:] = self.real_B[i, :, height_diff//2:self.x1[i], :]
            ct_bottom = torch.zeros_like(single_image)
            ct_bottom[0,:,x_bottom:,:] = self.real_B[i, :, self.x2[i]:self.x2[i]+256-x_bottom, :]
            interpolated_image = single_image+ct_upper+ct_bottom
            #interpolated_image = F.interpolate(single_image, size=(height, single_image.shape[3]), mode='bilinear', align_corners=False)
            x_stage1_list.append(interpolated_image)


        self.fake_B = torch.cat(fake_B_raw_list, dim=0)
        self.fake_B_coarse = torch.cat(x_stage1_list, dim=0)
        
        mask_center = torch.zeros_like(self.mask)
        width,length = mask_center.shape[2:]
        center_length = length//2

        mask_center[:,:,:,center_length-35:center_length+35]=1
        self.fake_B_local = self.mask*self.fake_B*mask_center
        self.real_B_local = self.mask*self.real_B*mask_center
        

        self.real_edges = self.sobel_edge(self.real_B_mask)
        self.fake_edges = self.sobel_edge(self.fake_B_mask_raw)
        

    def backward_D_1(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        #fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        
        # 还可以做一个mask的分割任务，用来监督mask区域的生成效果
        #pred_fake,fake_mask_sigmoid = self.netD_1(self.fake_B.detach())
        pred_fake = self.netD_1(self.fake_B.detach())
        #self.fake_mask_raw = torch.where(fake_mask_sigmoid>0.5,torch.ones_like(fake_mask_sigmoid),torch.zeros_like(fake_mask_sigmoid))

        self.loss_D_fake_1 = self.criterionGAN(pred_fake, False)
        # Real
        #real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD_1(self.real_B)
        self.loss_D_real_1 = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D_1 = (self.loss_D_fake_1+ self.loss_D_real_1) * 0.5
        #self.loss_D_Dice_mask = 1-diceCoeff(fake_mask_sigmoid,self.mask,activation='none')
        #self.loss_D_1+=self.loss_D_Dice_mask
        self.loss_D_1.backward()
        
    def backward_D_2(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        #fake_AB = torch.cat((self.real_A_mask, self.fake_B_mask), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD_2(self.fake_B_mask_raw.detach())
        self.loss_D_fake_2 = self.criterionGAN(pred_fake, False)
        # Real
        #real_AB = self.real_B_mask
        pred_real = self.netD_2(self.real_B_mask)
        self.loss_D_real_2 = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D_2 = (self.loss_D_fake_2 + self.loss_D_real_2) * 0.5
        self.loss_D_2.backward()
        
    def backward_D_3(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        #fake_AB = torch.cat((self.real_A, self.fake_B_local), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD_3(self.fake_B_local.detach())
        self.loss_D_fake_3 = self.criterionGAN(pred_fake, False)
        # Real
        #real_AB = torch.cat((self.real_A, self.real_B_local), 1)
        pred_real = self.netD_3(self.real_B_local)
        self.loss_D_real_3 = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D_3 = (self.loss_D_fake_3+ self.loss_D_real_3) * 0.5
        self.loss_D_3.backward()


    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        #fake_AB_ct = torch.cat((self.real_A, self.fake_B), 1)
        #pred_fake_ct,fake_mask_sigmoid = self.netD_1(self.fake_B)
        pred_fake_ct = self.netD_1(self.fake_B)
        #fake_AB_mask = torch.cat((self.real_A_mask, self.fake_B_mask), 1)
        pred_fake_mask = self.netD_2(self.fake_B_mask_raw)
        #fake_AB_ct_local = torch.cat((self.real_A, self.fake_B_local), 1)
        pred_fake_ct_local = self.netD_3(self.fake_B_local)
        #fake_AB_mask_local = torch.cat((self.real_A_mask, self.fake_B_mask_local), 1)

        #fake_edges = torch.cat((self.real_A, self.fake_edges), 1)

        self.loss_G_GAN = (self.criterionGAN(pred_fake_ct, True) + self.criterionGAN(pred_fake_mask, True) \
            + self.criterionGAN(pred_fake_ct_local, True) )/6
        # Second, G(A) = B
        
        #为了增加L1损失的大小，结合实际mask区域的大小而不是使用对整幅图像的差值绝对值的平均，因为mask只占整幅图像的一小部分
        mask_non_zero_count = torch.count_nonzero(self.mask)
        self.loss_G_maskL1 = (self.criterionL1(self.fake_B, self.real_B) + self.criterionL1(self.fake_B_coarse, self.real_B)) * 0.5 * \
            self.opt.lambda_L1 * (self.mask.shape[-1] * self.mask.shape[-1]  / mask_non_zero_count) *2

        
        #self.loss_G_nonmaskL1 = (self.criterionL1(self.fake_B_raw*(1.-self.mask), self.real_B*(1.-self.mask)) + self.criterionL1(self.x_stage1*(1.-self.mask), self.real_B*(1.-self.mask))) * 0.5 * \
        #    self.opt.lambda_L1
        # Third, Dice loss for mask
        self.loss_coarse_Dice = (1-diceCoeff(self.coarse_seg_sigmoid,self.normal_vert,activation='none'))*10
        #self.loss_G_Dice = (1-diceCoeff(self.fake_B_mask_sigmoid*self.mask,self.real_B_mask*self.mask,activation='none'))*10
        self.loss_G_Dice = (1-diceCoeff(self.fake_B_mask_sigmoid,self.real_B_mask,activation='none'))*15
        #self.loss_G_Dice_mask = diceCoeff(fake_mask_sigmoid.detach(),self.mask,activation='none')
        
        self.loss_edge = self.edge_loss(self.fake_edges, self.real_edges,reduction='mean')  * 800
        self.loss_h = torch.mean((abs(self.pred1_h-self.height)/self.height)*40+(abs(self.pred2_h-self.height)/self.height)*40)
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_maskL1 + self.loss_G_Dice+self.loss_edge+\
            self.loss_coarse_Dice + self.loss_h
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD_1, True)  # enable backprop for D
        self.optimizer_D_1.zero_grad()     # set D's gradients to zero
        self.backward_D_1()                # calculate gradients for D
        self.optimizer_D_1.step()          # update D's weights
        
        self.set_requires_grad(self.netD_2, True)  # enable backprop for D
        self.optimizer_D_2.zero_grad()     # set D's gradients to zero
        self.backward_D_2()                # calculate gradients for D
        self.optimizer_D_2.step()          # update D's weights
        
        self.set_requires_grad(self.netD_3, True)  # enable backprop for D
        self.optimizer_D_3.zero_grad()     # set D's gradients to zero
        self.backward_D_3()                # calculate gradients for D
        self.optimizer_D_3.step()          # update D's weights
        

        # update G
        self.set_requires_grad(self.netD_1, False)  # D requires no gradients when optimizing G
        self.set_requires_grad(self.netD_2, False)  # D requires no gradients when optimizing G
        self.set_requires_grad(self.netD_3, False)  # D requires no gradients when optimizing G

        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # update G's weights
