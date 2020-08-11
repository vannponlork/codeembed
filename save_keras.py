import tensorflow as tf
from keras.models import load_model
from keras import backend as K
from keras.applications import VGG19
import Utils_model
from Utils_model import VGG_LOSS
from keras.applications.vgg19 import VGG19
from keras.models import Model
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import utils
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import     build_signature_def, predict_signature_def
from tensorflow.contrib.session_bundle import exporter


image_height = 384
image_width = 384
image_h_lr = image_height /4
image_w_lr = image_width /4
amount = 2
batch_size = 2
image_folder = './train2019'
model_path = './models/model_4302/gen_model4302.h5'
export_path = './folder_to_export'
image_shape = (image_height,image_width, 3)
class VGG_LOSS(object):
def __init__(self, image_shape):
self.image_shape = image_shape
# computes VGG loss or content loss
def vgg_loss(self, y_true, y_pred):
vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=self.image_shape)
vgg19.trainable = False
# Make trainable as False
for l in vgg19.layers:
l.trainable = False
model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
model.trainable = False
return K.mean(K.square(model(y_true) - model(y_pred)))
loss = VGG_LOSS(image_shape)
image_shape = (image_height,image_width, 3)
K.set_learning_phase(0)
keras_model = load_model(model_path, custom_objects={'vgg_loss': loss.vgg_loss})
# Define num_output. If you have multiple outputs, change this number accordingly
config = keras_model.get_config()
weights = keras_model.get_weights()
new_model = keras_model.from_config(config)
new_model.set_weights(weights)
print(new_model.input)
print("="*100)
print(new_model.output)
builder = saved_model_builder.SavedModelBuilder(export_path)
signature = predict_signature_def(inputs={'images': new_model.input},
outputs={'scores': new_model.output})
with K.get_session() as sess:
builder.add_meta_graph_and_variables(sess=sess,
tags=[tag_constants.SERVING],
signature_def_map={'predict': signature})
builder.save()