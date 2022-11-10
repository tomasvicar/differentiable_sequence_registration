import numpy as np
from skimage.transform import rescale
from skimage.filters import gaussian
import torch


class create_masks_hamming():
    
    def __init__(self,crop_fraction,np_dtype):
        self.crop_fraction = crop_fraction
        self.np_dtype = np_dtype
        
    def __call__(self,shape):
    

        mask = np.hamming(shape[1]).reshape(-1,1) @ np.hamming(shape[2]).reshape(1,-1).astype(self.np_dtype)
        
        border = self.crop_fraction
        maskx = np.zeros(shape[1:]).astype( self.np_dtype)
        maskx[int(shape[1] * border):int(shape[1] * (1-border)),int(shape[2] * border):int(shape[2] * (1-border))] = 1
        
        mask[maskx == 0] = 0
    
        return np.tile(mask.reshape((1, shape[1], shape[2])), (shape[0], 1, 1))
    
        
        
class create_masks_crop():
    
    def __init__(self,crop_fraction, np_dtype):
        self.crop_fraction = crop_fraction
        self.np_dtype = np_dtype
        
    def __call__(self,shape):
        
        border = self.crop_fraction
        mask = np.zeros(shape[1:]).astype(self.np_dtype)
        mask[int(shape[1] * border):int(shape[1] * (1-border)),int(shape[2] * border):int(shape[2] * (1-border))] = 1
        
        
        return np.tile(mask.reshape((1, shape[1], shape[2])), (shape[0], 1, 1))

    
    
def padding(imgs, masks, pad, padval=0):
    if pad > 0:
        imgs = np.pad(imgs, [0, pad, pad], constant_values=padval)
        masks = np.pad(masks, [0, pad, pad], constant_values=padval)
        
    return imgs, masks
    

def resize_pyramid(imgs, masks, scale, sigma, np_dtype):
    
    imgs_res = []
    masks_res = []
    for img, mask in zip(imgs,masks):
        if sigma > 0:
            img = gaussian(img,sigma)
            
        imgs_res.append(rescale(img, 1/scale, preserve_range=True).astype(np_dtype))
        masks_res.append(rescale(mask, 1/scale, preserve_range=True).astype(np_dtype))
        
    return np.stack(imgs_res,0), np.stack(masks_res,0) 
        
        
class Randomizer():
    
    def __init__(self, num_batches, num_imgs):
        self.num_batches = num_batches
        self.num_imgs = num_imgs
        self.rand_inds = np.random.permutation(num_imgs)
        self.batch_num = 0
        
    
    def get_batch_inds(self, batch_num):
        ind_start = (self.num_imgs //  self.num_batches) * batch_num
        ind_stop = (self.num_imgs //  self.num_batches) * (batch_num+1)
        if batch_num == (self.num_batches - 1):
            ind_stop = self.num_imgs 
        inds = self.rand_inds[ind_start:ind_stop]
        
        return inds
        
    def __iter__(self):
        return self
        
    def __next__(self):
        self.batch_num  += 1
        
        if self.batch_num  == (self.num_batches + 1):
            raise StopIteration
            
        inds =  self.get_batch_inds(self.batch_num - 1)
        
        return inds
        
    def order_corection(self,output):
    
        order_corection = np.argsort(self.rand_inds)
        output = output[order_corection,:,:]
        
        return output
        
        
        
def construct_matrix(params, inds, device):
    
    angle_rad, tx, ty = params
    
    theta = torch.zeros((len(inds)), 3, 3).to(device)
    theta[:,2,2] = 1

    theta[:,0,0] = 1 * torch.cos(angle_rad[inds])
    theta[:,1,1] = 1 * torch.cos(angle_rad[inds])
    theta[:,0,1] = torch.sin(angle_rad[inds])
    theta[:,1,0] = -torch.sin(angle_rad[inds])
    
    theta[:,0,2] = tx[inds]
    theta[:,1,2] = ty[inds]
    
    return theta
    

        
        
        
        
        
        
        
        
            
            