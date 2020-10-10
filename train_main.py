
import os
from dataloader import get_doc_loaders
from train import train
from tensorflow.keras.applications import vgg16
import numpy as np
import tensorflow as tf



def vgg_preprocessing(input_data):
    return vgg16.preprocess_input(np.copy(input_data.astype('float32')))

def make_experiment(name, ref_path, target_path, alien_path, epoch_num, first_trained_layer_name, lambda_, output_dir, batchsize, size, split_val, cls_num, preprocessing_func, network_constractor):
  output_dir = os.path.join(output_dir, name)
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  with open(os.path.join(output_dir, "experiment_settings.txt"), "w") as set_file:
      settings = [name, ref_path, target_path, alien_path, epoch_num, first_trained_layer_name, lambda_, output_dir, batchsize, size, split_val, cls_num, preprocessing_func, network_constractor]
      for item in settings:
        set_file.write(str(item) +"\n")

  train_s_loader, ref_loader, test_s_loader, test_b_loader = get_doc_loaders(ref_path, target_path, alien_path, batchsize, (size,size), split_val, cls_num, shuffle=True, preprocess_func=preprocessing_func)

  train_s_loader.write_data(output_dir, "target_train")
  test_s_loader.write_data(output_dir, "target_test")
  test_b_loader.write_data(output_dir, "alien_test")

  return train(train_s_loader, ref_loader, epoch_num, first_trained_layer_name, lambda_, output_dir, network_constractor, batchsize)



size = 224
preprocessing_func = vgg_preprocessing
first_trained_layer_name =  "block5_conv1"
network_constractor = lambda : tf.keras.applications.VGG16(include_top=True, input_shape=(size, size, 3), weights='imagenet')

REFERENCE_PATH = "C:\\Users\\lotan\\Documents\\studies\\Affordances\\datasets\\imagenet_val_splitted"
ALIEN_PATH = "C:\\Users\\lotan\\Documents\\studies\\Affordances\\datasets\\clean_alien_visulization"
OUTPUT_PATH = "C:\\Users\\lotan\\Documents\\studies\\Affordances\\experiments"


cls_num = 1000
batchsize = 2
split_val = 0.2

NAME = "experiment_unclean_target_all_stab"
TARGET_PATH = "C:\\Users\\lotan\\Documents\\studies\\Affordances\\datasets\\data"
lambda_ = 1

if __name__ == "__main__":
    model_t = make_experiment(NAME, REFERENCE_PATH, TARGET_PATH, ALIEN_PATH, 5, first_trained_layer_name, lambda_, OUTPUT_PATH, batchsize, size, split_val, cls_num, preprocessing_func, network_constractor)
