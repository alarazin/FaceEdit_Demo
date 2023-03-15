import tensorflow as tf 
import numpy as np 
import matplotlib.pylab as plt
import sys
sys.path.append("../stylegan-encoder")
import dnnlib.tflib as tflib

class LocalOptimizer:
    def __init__(self, local_editor, w_edit, y_edit, y_mask, y_edit_mask, h_edit_1024, edit_dict):
        self.l_editor = local_editor
        self.l_editor.layers[1].set_weights([w_edit.reshape((-1,18*512))])
        train_data = tf.constant(np.arange(1)[np.newaxis, :])
        im_original = self.l_editor(train_data)
        scale = 255/2
        im_original = im_original * scale + (0.5+scale)
        im_original = tf.clip_by_value(im_original, clip_value_min = 0, clip_value_max = 255)/255
        self.im_original =tf.math.multiply(tf.constant(y_edit_mask, dtype = tf.float32), im_original)
        self.y_edit = y_edit 
        self.im_edit = y_edit_mask*self.y_edit
        self.y_mask = y_mask
        self.im_mask =y_edit_mask*tflib.tfutil.convert_images_to_uint8(self.y_mask).eval()/255
        self.h_edit_1024 = h_edit_1024
        self.h_edit_1024_tf = tf.constant(h_edit_1024, dtype =tf.float32)
        self.edit_dict = edit_dict
    def set_masks(self, au_local_list):
        
        #au_local_list=[[14,15,25], [13,14,15,17,25], [14,15,25], [15], [1,15,21,27,29], [15], [10,21],[8,21], [1,8,11, 21,23,24,29],[1,8,11,21,23,24],[1,8,11,21,23,24],[8,11,24,23],[1,8,11,21,23,24],[1,8,11,21,23,24], [7,8,11,21,24], [7,8,11,21,24],[]]
        #au_local_list = [[9, 15], [13, 0, 9, 18, 8, 15], [15, 9,18,8,0,16], [16], [16, 3, 1, 19, 21, 14,6], [16], [2, 17, 20, 6, 1,3], [20,6,10,1], [1,6,10,14,19,5], [1,6,10,14,19,5], [6, 10,5,14,19], [20,6,10,5,14,12],[6,10,5,14,19,21], [6,10,5,14], [10,5,14], [10,5,14,12,19]]
        #au_local_list = [[9, 15], [11,13, 0, 9, 18, 8, 15], [15, 9], [16], [16, 3, 1, 6, 19, 21], [16], [2, 17, 20, 6, 1, 10], [20,6,10,1], [1,6,10,14,19,5,21,3], [1,6,10,14,19,5], [6, 10,5,14,19], [20,6,10,5,14,12],[6,10,5,14,19,21], [6,10,5,14], [10,5,14], [10,5,14,12,19]]

        already_edited=[]
        all_subtract_original=[]
        all_subtract_edit = []
        all_subtract_mask = []
        
        for au in self.edit_dict:
            for k in au_local_list[au]:
                if k not in already_edited:
                    already_edited.append(k) 
                    subtract_original = []
                    subtract_edit=[]
                    subtract_mask = []
                    
                    for i in range(3):
                        subtract_original.append(tf.math.multiply(self.h_edit_1024_tf[:,:,k], self.im_original[0][i,:,:]))
                        subtract_edit.append(np.multiply(self.h_edit_1024[:,:,k], self.im_edit[0][i,:,:]))
                        subtract_mask.append(np.multiply(self.h_edit_1024[:,:,k], self.im_mask[0][i,:,:]))
                    
                    subtract_original_tensor = tf.reshape(tf.stack(subtract_original), [1, 3, 1024, 1024])
                    all_subtract_original.append(subtract_original_tensor)
      
                    subtract_edit_array = np.array(subtract_edit).reshape((1,3,1024, 1024))
                    all_subtract_edit.append(subtract_edit_array)

                    subtract_mask_array = np.array(subtract_mask).reshape((1,3,1024, 1024))
                    all_subtract_mask.append(subtract_mask_array)
                    
        self.mask_original = tf.add_n(all_subtract_original)
        mask_edit = sum(all_subtract_edit)
        self.mask_mask = sum(all_subtract_mask)
        
        self.im_edit2 = self.im_edit - mask_edit
        self.im_pred = tf.math.subtract(self.im_original, tf.add_n(all_subtract_original))
        
    def losses(self):
        mse = tf.keras.losses.MeanSquaredError()
        loss1 = mse(self.im_edit2, self.im_pred)
        loss2 = mse(self.mask_mask, self.mask_original)
        self.loss = (loss1+loss2)/2
        
    def optimize(self, iterations):
        optimizer = tf.train.AdamOptimizer()
        min_op = optimizer.minimize(self.loss, var_list = [self.l_editor.trainable_weights])
        
        sess = tf.get_default_session()
        sess.run(tf.variables_initializer(optimizer.variables()))
        fetch_ops = [min_op, self.loss]
        for _ in range(iterations):
            _, self.loss = sess.run(fetch_ops)
        
        
        
    def plot_masks(self):
        f, axarr = plt.subplots(1,3,figsize = (50,100))
        axarr[0].imshow(self.mask_original.eval()[0].transpose((1,2,0)))
        axarr[1].imshow(self.im_mask[0].transpose((1,2,0)))
        axarr[2].imshow(self.im_edit2[0].transpose((1,2,0)))
        plt.show()
        
    def get_result(self):
        return self.l_editor.layers[1].get_weights()[0]
       
        
        
        
        
        
        
                        
