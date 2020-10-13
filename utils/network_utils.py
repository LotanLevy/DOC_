
from tensorflow.keras.applications import vgg16
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model


def get_network_functions(network_name, cls_num, input_size):
    if network_name == "vgg" and cls_num == 1000:
        vgg_preprocessing = lambda input_data: vgg16.preprocess_input(np.copy(input_data.astype('float32')))
        vgg_constructor = lambda : tf.keras.applications.VGG16(include_top=True, input_shape=(input_size, input_size, 3), weights='imagenet')
        return vgg_preprocessing, vgg_constructor
    elif network_name == "vgg":
        vgg_preprocessing = lambda input_data: vgg16.preprocess_input(np.copy(input_data.astype('float32')))
        def model_constructor():
            model = tf.keras.applications.VGG16(include_top=True, input_shape=(input_size, input_size, 3), weights='imagenet')
            # features_model = Model(inputs=model.input, outputs=model.layers[-2].output)
            prediction = Dense(cls_num, activation='softmax')(model.layers[-2].output)

            return Model(inputs=model.input, outputs=prediction)
        return vgg_preprocessing, model_constructor


