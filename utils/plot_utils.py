import tensorflow as tf 
import matplotlib.pylab as plt
import sys
sys.path.append("../stylegan-encoder")
import dnnlib.tflib as tflib

from img_utils import mask_img

def plot_final_result(y_edit, y_mask, im_local, im_id,latent_dirs, path_align, mask_path):
    f, axarr = plt.subplots(1,3,figsize = (50,100))
    axarr[0].imshow(y_edit[0].transpose((1,2,0)))
    axarr[1].imshow(mask_img(im_id, tflib.tfutil.convert_images_to_uint8(y_mask[0].transpose((1,2,0))).eval(), latent_dirs, path_align, mask_path))
    axarr[2].imshow(mask_img(im_id, im_local[0], latent_dirs, path_align, mask_path))
plt.show()