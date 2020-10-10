

import os
import datetime
import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Model
from dataloader import construct_with_files
from train_main import vgg_preprocessing
from test_utils.ROC_graph import get_data_scores, get_roc_curve
from test_utils.scores_graphs import create_images_graph
from test_utils.features_graphs import get_features_graph
from test_utils.CAM_grads_visulaization import get_results_for_imagesdir
from experiment_utils import Experiment


# NAME= "experiment_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
NAME = "experiment_test"
OUTPUT_MAIN_DIR = "C:\\Users\\lotan\\Documents\\studies\\Affordances\\experiments"
EPOCH_NUM = 5
EXPERIMENT2DIR = {"all_stab_model": "C:\\Users\\lotan\\Documents\\studies\\Affordances\\experiments\\experiment_unclean_target_all_stab",
                     "knives_only_model" : "C:\\Users\\lotan\Documents\\studies\\Affordances\\experiments\\experiment_unclean_target_only_knives\\20201007-141227",
                  "all_stab_with_lambd01" : "C:\\Users\\lotan\\Documents\\studies\\Affordances\\experiments\\experiment_unclean_target_all_stab_lamd01\\20201007-155621",
                    "knives_only_with_lambd01" : "C:\\Users\\lotan\\Documents\\studies\\Affordances\\experiments\\experiment_unclean_target_only_knives_lamd01\\20201007-200628",
                  "untrainbed_model_with_knives_data": "C:\\Users\\lotan\\Documents\\studies\\Affordances\\experiments\\untrained_with_knives_data"}


OUTPUT_PATH =  os.path.join(OUTPUT_MAIN_DIR, NAME)

if not os.path.exists(OUTPUT_PATH):
  os.makedirs(OUTPUT_PATH)

size = 224
classes_num = 1000
preprocessing_func = vgg_preprocessing
network_constractor = lambda : tf.keras.applications.VGG16(include_top=True, input_shape=(size, size, 3), weights='imagenet')




# creates_relevant_models
experiments = dict()

for model_name in EXPERIMENT2DIR:
    experiments[model_name] = Experiment(model_name, EXPERIMENT2DIR[model_name], network_constractor, EPOCH_NUM, preprocessing_func)



# Creates epoch dirs
get_epoch_dir = lambda model_name, epoch: os.path.join(os.path.join(OUTPUT_PATH, model_name), "epoch_{}".format(epoch))
for model_name in experiments:
  model_dir = os.path.join(OUTPUT_PATH, model_name)
  if not os.path.exists(get_epoch_dir(model_dir, EPOCH_NUM)):
    os.makedirs(get_epoch_dir(model_dir, EPOCH_NUM))

#
# # Creates roc curve graph
# for name, experiment in experiments.items():
#   get_roc_curve(experiment.model, experiment.templates, experiment.target, experiment.aliens, get_epoch_dir(name, EPOCH_NUM))
#

norm_factors = dict()
z2s = dict()

# Creates scores graphs
for name, experiment in experiments.items():
  Z1 = get_data_scores(experiment.model, experiment.templates, experiment.target)
  Z2 = get_data_scores(experiment.model, experiment.templates, experiment.aliens)

  norm_factors[name] = max(np.max(Z1), np.max(Z2))
#   z2s[name] = Z2
#   # Z1 /= norm_factors[model_name]
#   # Z2 /= norm_factors[model_name]
#   scores_graph_output_path = get_epoch_dir(name, EPOCH_NUM)
#   create_images_graph(scores_graph_output_path, experiment.target_paths[:40], Z1[:40], "scores_for_knives_images", 0.08, 20)  # displays the first 40's examples
#   create_images_graph(scores_graph_output_path, experiment.target_paths, Z1, "the_smallest_scores_for_knives_images", 0.08, 20, 20) # the 20's examples with the lowest score
#   create_images_graph(scores_graph_output_path, experiment.aliens_paths[:40], Z2[:40], "scores_for_alien_images", 0.08, 20) # displays the first 40's examples
#   create_images_graph(scores_graph_output_path, experiment.aliens_paths, Z2, "the_smallest_scores_for_alien_images", 0.05, 20, 20) # the 20's examples with the lowest score
# #
# for name, experiment in experiments.items():
#   features_graph_output_path = get_epoch_dir(name, EPOCH_NUM)
#   get_features_graph(experiment.model, experiment.templates, experiment.target, experiment.aliens, features_graph_output_path)

grad_cam_output_path = os.path.join(os.path.join(OUTPUT_PATH, "grad_cam"), "epoch_" + str(EPOCH_NUM))
if not os.path.exists(grad_cam_output_path):
    os.makedirs(grad_cam_output_path)
to_visualize_images = list(experiments.values())[0].aliens
to_visualize_paths = list(experiments.values())[0].aliens_paths

losses = get_results_for_imagesdir(experiments, to_visualize_images, to_visualize_paths, grad_cam_output_path,
                                   norm_factors)