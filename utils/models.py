import tensorflow as tf

import keras
from keras.models import load_model, Sequential, Model
from keras.layers import Embedding, Flatten




class Stylegan(keras.layers.Layer):
  def __init__(self, generator):
    super(Stylegan, self).__init__()
    self.sess = tf.get_default_session()
    self.graph = tf.get_default_graph()
    self.generator = generator
  
  def call(self, inputs):
    images_out= self.generator.get_output_for(tf.reshape(inputs, [-1,18,512]), randomize_noise = False)
    x= tf.identity(images_out, name = 'images_out')
    return x
  def compute_output_shape(self, input_shape):
    return (None, 3, 1024, 1024)




class EditorModel:
  def __init__(self):
    self.sess = tf.get_default_session()
    #self.gan = Stylegan(Gs)
  
  def global_editor(self, m_combined, size):
    #m_combined=load_model(os.path.join('models', 'combined_model_linear'))

    embedding=Sequential(name='Embedding')
    embedding.add(Embedding(size,18*512, input_length=1))
    embedding.add(Flatten())

    nonlinear_comb=Model(embedding.input, m_combined(embedding.output))
    nonlinear_comb.layers[-1].trainable=False
    nonlinear_comb.compile(optimizer = 'adam', loss= 'mse')
    #nonlinear_comb.layers[1].set_weights([w[0].reshape((-1,18*512))])
    #self.y_real=nonlinear_comb.predict(np.arange(1))
    return nonlinear_comb

  def local_editor(self,Gs):
    
    gan = Stylegan(Gs)
    editor=Sequential(name='Editor')
    editor.add(Embedding(1,18*512, input_length=1, input_shape = (1,)))

    trial = Model(editor.input, gan(editor.output))
    trial.layers[-1].trainable = False
    #trial.compile('adam', 'mse')
    return trial



