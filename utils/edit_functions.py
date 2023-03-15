import numpy as np 

def get_layer_activations(w, Gs, conv_name = '32x32/Conv1'): 
    w=w.reshape((1,18,-1))
    output=[]
  
    for layer_name, layer_output, layer_trainables in Gs.list_layers():
        if 'dlatents_in' in layer_name:
            output.append(layer_output)   
        if conv_name==layer_name:
            conv_activations=layer_output.eval(feed_dict={output[0].name: w})
            output.append(layer_output)
            break
    return conv_activations, output

def transfer_activations_local(Gs,original_conv_activations, target_conv_activations, au, heatmaps, au_local_dict, combine = None):
    conv_activations_edit = original_conv_activations.copy()
    h =heatmaps._data[32][0]
    local_h = au_local_dict[au]
    for i in local_h:
        target_local = np.multiply(h[i], target_conv_activations[7][0])
        conv_activations_edit[7][0] = conv_activations_edit[7][0]-(np.multiply(h[i], conv_activations_edit[7][0]))+target_local
        
    return conv_activations_edit

def transfer_activations_local2(Gs,original_conv_activations, target_conv_activations, heatmaps, au, au_local_dict, combine = None, layer_num = 7):
    conv_activations_edit = original_conv_activations.copy()
    k_list = [2,4,6, 9, 10, 11, 13,14, 15, 17, 18,19, 22, 24, 27, 28, 30, 31, 33, 34, 35, 36]
    h =heatmaps._data[32][0][k_list]
    local_h = au_local_dict[au]
    for i in local_h:
        target_local = np.multiply(h[i], target_conv_activations[0])
        conv_activations_edit[0] = conv_activations_edit[0]-(np.multiply(h[i], conv_activations_edit[0]))+target_local
        
    return conv_activations_edit

def edit_activations(w,Gs, conv_activations_edit, output, layer_list):
    w = w.reshape((1,18,-1))
    conv_edit = []
    rgb_edit = []
    all_edit= []
    
    conv_activations_edit.insert(0,w)
    for layer_name, layer_output, layer_trainables in Gs.list_layers():
        layer_eval = layer_output.eval(feed_dict={output[i].name: conv_activations_edit[i] for i in layer_list})
        if 'Conv' in layer_name or 'Const' in layer_name:
            conv_edit.append(layer_eval)
        elif 'RGB' in layer_name or 'images_out' in layer_name:
            rgb_edit.append(layer_eval)
            
    return conv_edit, rgb_edit

def edit_activations2(w,Gs,conv_activations_edit,output):
    w=w.reshape((1,18,-1))
    
    conv_activations_edit.insert(0,w)
    for layer_name, layer_output, layer_trainables in Gs.list_layers():
        layer_eval = layer_output.eval(feed_dict={output[i].name: conv_activations_edit[i] for i in range(2)})
        if 'images_out' in layer_name:
            rgb_edit = layer_eval
    
    return rgb_edit
    



def get_activations(w,Gs):    #function to get activations. w must be reshaped ((1,18,-1))
  w=w.reshape((1,18,-1))
  conv_activations=[]
  rgb_activations=[]
  all_activations=[]
  output=[]

  for layer_name, layer_output, layer_trainables in Gs.list_layers():
    if 'dlatents_in' in layer_name:
      output.append(layer_output)
    
    layer_eval=layer_output.eval(feed_dict={output[0].name: w})

    if 'Conv' in layer_name or 'Const' in layer_name:
      conv_activations.append(layer_eval)
      output.append(layer_output)
    elif 'RGB' in layer_name or 'images_out' in layer_name:
      rgb_activations.append(layer_eval)

    all_activations.append(layer_eval)

  return conv_activations, rgb_activations, all_activations, output    




def get_directions(model):
    dirs=[]
    for i in range(1,len(model.layers)):
        dirs.append(model.layers[i].get_weights())
    return dirs


def edit_w_directions(edit_dict, w_original, directions, global_intensity, lower_layer=2, upper_layer=8):
    w_edit = w_original.copy()
    edit_dirs = np.zeros((1024,1))
    for au in edit_dict:
        edit_dirs+= edit_dict[au]*directions[au+2][0]@directions[au+2][2]@directions[au+2][4]
    
    w_edit[lower_layer:upper_layer] = (w_original+ global_intensity*((directions[0][0])@(directions[1][0])@edit_dirs).reshape((18,512)))[lower_layer:upper_layer]
    
    return w_edit

def transfer_to_original(w_original, w_edit, lmin = 2, lmax= 8):
    w_final = w_original[0].copy()
    w_final[lmin:lmax] = w_edit.reshape((18,512))[lmin:lmax]
    return w_final.reshape((1,18,512))

