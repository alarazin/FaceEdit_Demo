import PIL.Image
from PIL import ImageFilter
import os
import numpy as np 

def mask_img(im_id,im,latents, path_align, mask_path):
    #latents_dir = sorted(os.listdir(latents))
    
    im_name=latents[im_id][:-4]+'.png'
    if not os.path.exists(os.path.join(mask_path, im_name)):
        return im
    else:
        width,height=im.shape[0], im.shape[1]
        orig_img=PIL.Image.open(os.path.join(path_align, im_name)).convert('RGB').resize((width, height))
        imask=PIL.Image.open(os.path.join(mask_path, im_name)).convert('L').resize((width,height))
        imask=imask.filter(ImageFilter.GaussianBlur(15))
        mask=np.array(imask)/255
        mask=np.expand_dims(mask, axis=-1)
        img_array=mask*np.array(im)+(1.0-mask)*np.array(orig_img)
        img_array=img_array.astype(np.uint8)
        #return PIL.Image.fromarray(img_array, 'RGB')
        return img_array

def mask_img2(im, img_path, mask_path):
    if not os.path.exists(mask_path):
        mask_path = os.path.join('masks', 'photo_23-Jun-2020 (13_44_57.510968)_01.png')
    width,height=im.shape[0], im.shape[1]
    orig_img=PIL.Image.open(img_path).convert('RGB').resize((width, height))
    imask=PIL.Image.open(mask_path).convert('L').resize((width,height))
    imask=imask.filter(ImageFilter.GaussianBlur(10))
    mask=np.array(imask)/255
    mask=np.expand_dims(mask, axis=-1)
    img_array=mask*np.array(im)+(1.0-mask)*np.array(orig_img)
    img_array=img_array.astype(np.uint8)
    return img_array #PIL.Image.fromarray(img_array, 'RGB')

    

def load_img(im_id, latents, lantent_dirs):
    return np.load(os.path.join(latents, latent_dirs[im_id]))[np.newaxis,:]

def load_mask(im_id, latent_dirs, mask_path):
    y_edit_mask = np.asarray(PIL.Image.open(os.path.join(mask_path, latent_dirs[im_id][:-3]+'png')).resize((1024, 1024)))
    return np.concatenate([y_edit_mask[np.newaxis,...] for i in range(3)], axis = 0)/255

def prep_im(rgb, im_range = 255):
    im = rgb[0].transpose((1,2,0))
    im = im-np.min(im)
    im = im/np.max(im)
    im = im*im_range

    return im
    