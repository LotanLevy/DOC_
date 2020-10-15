
import argparse
import os
import numpy as np
from test_utils.ROC_graph import get_roc_curve
from test_utils.scores_graphs import create_images_graph
from test_utils.CAM_grads_visualization import get_results_for_imagesdir
from utils.experiment_utils import Experiment
import datetime




def get_test_parser():
    parser = argparse.ArgumentParser(description='Process test args.')

    parser.add_argument('--models_dirs', type=str, action='append', required=True, help='<Required> Set flag')
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--name', type=str, default=datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    parser.add_argument('--epochs_weights_num', type=int, default=5)
    parser.add_argument('--templates_num', type=int, default=40)
    parser.add_argument('--target_num', type=int, default=100)
    parser.add_argument('--alien_num', type=int, default=100)


    parser.add_argument('--target2alien_roc', action='store_true')
    parser.add_argument('--alien2alien_roc', action='store_true')
    parser.add_argument('--scores_graph', action='store_true')
    parser.add_argument('--features_graph', action='store_true')
    parser.add_argument('--cam_grads_images', action='store_true')
    parser.add_argument('--creative_scores', action='store_true')
    return parser

def run_test(args):
    output_path = os.path.join(args.output_path, args.name)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # creates_relevant_models
    experiments = dict()
    norm_factors = dict()
    get_epoch_dir = lambda model_name, epoch: os.path.join(os.path.join(output_path, model_name), "epoch_{}".format(epoch))

    for model_path in args.models_dirs:
        new_experiment = Experiment(model_path, args.epochs_weights_num)
        experiments[new_experiment.experiment_name] = new_experiment
        # Creates models output dir
        model_and_epoch_path = get_epoch_dir(new_experiment.experiment_name, args.epochs_weights_num)
        if not os.path.exists(model_and_epoch_path):
            os.makedirs(model_and_epoch_path)
        Z1 = new_experiment.get_data_scores(new_experiment.model, new_experiment.templates, new_experiment.target)
        Z2 = new_experiment.get_data_scores(new_experiment.model, new_experiment.templates, new_experiment.aliens)
        norm_factors[new_experiment.experiment_name] = max(np.max(Z1), np.max(Z2))

    print(norm_factors)

    # target2alien roc curve
    if args.target2alien_roc:
        for model_name, experiment in experiments.items():
            get_roc_curve(experiment.get_data_scores, "targets_to_aliens_for_"+model_name, experiment.model, experiment.templates, experiment.target, experiment.aliens,
                          get_epoch_dir(model_name, args.epochs_weights_num))

    # alien2alien roc curve
    if args.alien2alien_roc:
        for model_name, experiment in experiments.items():
            get_roc_curve(experiment.get_data_scores, "aliens_to_aliens_for_"+model_name, experiment.model, experiment.templates, experiment.aliens_positive, experiment.aliens_negative,
                          get_epoch_dir(model_name, args.epochs_weights_num))

    if args.scores_graph:
        # Creates scores graphs
        for name, experiment in experiments.items():
            Z1 = experiment.get_data_scores(experiment.model, experiment.templates, experiment.target)
            Z2 = experiment.get_data_scores(experiment.model, experiment.templates, experiment.aliens)
            scores_graph_output_path = get_epoch_dir(name, args.epochs_weights_num)
            create_images_graph(scores_graph_output_path, experiment.target_paths[:40], Z1[:40],
                                "scores_for_target_images_for_"+name,  0.08, 20)  # displays the first 40's examples
            create_images_graph(scores_graph_output_path, experiment.target_paths, Z1,
                                "the_smallest_scores_for_target_images_for_"+name, 0.08, 20,
                                20)  # the 20's examples with the lowest score
            create_images_graph(scores_graph_output_path, experiment.aliens_paths[:40], Z2[:40],
                                 "scores_for_alien_images_for_"+name, 0.08, 20)  # displays the first 40's examples
            create_images_graph(scores_graph_output_path, experiment.aliens_paths, Z2,
                                "the_smallest_scores_for_alien_images_for_"+name, 0.05, 20, 20)  # the 20's examples with the lowest score

    if args.cam_grads_images:
        grad_cam_output_path = os.path.join(os.path.join(output_path, "grad_cam"), "epoch_" + str(args.epochs_weights_num))
        if not os.path.exists(grad_cam_output_path):
            os.makedirs(grad_cam_output_path)
        to_visualize_images = list(experiments.values())[0].aliens
        to_visualize_paths = list(experiments.values())[0].aliens_paths
        get_results_for_imagesdir(experiments, to_visualize_images, to_visualize_paths,
                                           grad_cam_output_path,
                                           norm_factors)

if __name__ == "__main__":
    run_test(get_test_parser().parse_args())
