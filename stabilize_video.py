from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import os
from scipy.io import savemat


from config import Config
from video_utils import load_video
from video_utils import load_video_nocorrupted
from video_utils import save_video
from utils import padding
from utils import resize_pyramid
from utils import Randomizer
from utils import construct_matrix
from losses import weighted_MSE_valid
from losses import diagonal_regularization
from utils import create_masks_crop



def stabilize_video(file_name, config):
    

    imgs = load_video(file_name,config.np_dtype)
    
    masks = config.create_masks(imgs.shape)
    
    imgs,masks = padding(imgs,masks,config.pad)


    angle_rad = torch.zeros(imgs.shape[0]).to(config.device)
    tx = torch.zeros(imgs.shape[0]).to(config.device)
    ty = torch.zeros(imgs.shape[0]).to(config.device)
    angle_rad.requires_grad = True
    tx.requires_grad = True
    ty.requires_grad = True
    
    params = [angle_rad,tx,ty]


    losses = []
    for scale_num, (scale, sigma) in enumerate(zip(config.scales,config.sigmas)):
        
        imgs_res, masks_res = resize_pyramid(imgs, masks, scale, sigma, config.np_dtype)
        
        imgs_res = torch.unsqueeze(torch.from_numpy(imgs_res),1)
        masks_res = torch.unsqueeze(torch.from_numpy(masks_res),1)

        optimizer = torch.optim.Adam(params,lr=config.init_lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config.iterations, gamma=config.gamma, last_epoch=-1)

        
        for it in range(config.iterations[-1]):

            output = []
            loss_tmp = []
            randomizer = Randomizer(config.num_batches, imgs_res.shape[0])
            for batch_num, inds in enumerate(randomizer):
                
                imgs_res_batch = imgs_res[inds,:,:,:].to(config.device)
                masks_res_batch = masks_res[inds,:,:,:].to(config.device)
                
                theta = construct_matrix(params, inds, config.device)
                
                
                grid = F.affine_grid(theta[:,0:2,:], imgs_res_batch.shape, align_corners=config.align_corners)
                output_batch = F.grid_sample(imgs_res_batch, grid, padding_mode="zeros", align_corners=config.align_corners, mode=config.interp_mode)
                with torch.no_grad(): 
                    out_masks = F.grid_sample(masks_res_batch, grid, padding_mode="zeros", align_corners=config.align_corners, mode=config.interp_mode)
                
                
                loss1 = weighted_MSE_valid(output_batch,out_masks)
                loss2 = diagonal_regularization(theta, config.device)
                
                loss = loss1 + config.regularization_factor * loss2

                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                output.append(output_batch[:,0,:,:].detach().cpu().numpy())
                
                loss_tmp.append(loss.detach().cpu().numpy())
                
                
                
            output = np.concatenate(output,0)
            output= randomizer.order_corection(output)
            losses.append(np.mean(loss_tmp))
            
            if (it % 5) == 0:
                print(it)
                print(theta[0,:,:])
                print(loss1,loss2)
                
                plt.plot(losses)
                plt.show()
                
                std1 = np.std(imgs_res[:,0,:,:].detach().cpu().numpy(), axis=0)
                std2 = np.std(output, axis=0)   
                plt.imshow(np.concatenate( (std1, std2), axis=1))
                plt.show()
    

    
    file_name_save = file_name.replace('.avi', '')  + '_registered.avi'
    output = np.round(output*255).astype(np.uint8)
    save_video(file_name_save,output) ###FFMPEG isntalation is required https://www.geeksforgeeks.org/how-to-install-ffmpeg-on-windows/
    
        
    file_name_save = file_name.replace('.avi', '')  + '_transformations.npy'
    theta = construct_matrix(params, config.resize_factor, list(range(output.shape[0])), torch.device('cpu')).detach().cpu().numpy()
    np.save(file_name_save, theta)
               
         


    
if __name__ == "__main__":
    
    file_name = 'example_data.avi'
    config = Config()
    
    stabilize_video(file_name, config)
    
    
    
    
