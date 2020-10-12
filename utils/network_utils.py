
from tensorflow.keras.applications import vgg16
import numpy as np
import tensorflow as tf

def get_network_functions(network_name, cls_num, input_size):
    if network_name == "vgg" and cls_num == 1000:
        vgg_preprocessing = lambda input_data: vgg16.preprocess_input(np.copy(input_data.astype('float32')))
        vgg_constructor = lambda : tf.keras.applications.VGG16(include_top=True, input_shape=(input_size, input_size, 3), weights='imagenet')
        return vgg_preprocessing, vgg_constructor