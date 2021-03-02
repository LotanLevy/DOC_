

import argparse
import datetime
import os
from test_utils.ROC_graph import get_roc_curve
import numpy as np


# from utils.experiment_utils import Trainer
from utils import experiment_utils

def get_train_parser():
    parser = argparse.ArgumentParser(description='Process train args.')

    parser.add_argument('--ref_dir', type=str, required=True)
    parser.add_argument('--tar_dir', type=str, required=True)
    parser.add_argument('--alien_dir', type=str, required=True)
    parser.add_argument('--alien_cls2label', type=str, default=None)
    parser.add_argument('--use_aug', type=int, default=0, choices=[0,1])



    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--name', type=str, default=datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    parser.add_argument('--batchsize', '-bs', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--cls_num', type=int, default=1000)
    parser.add_argument('--reg_coeff', type=float, default=0.1)
    parser.add_argument('--reg_samples', type=int, default=500)

    parser.add_argument('--use_var_reg', type=int, default=0, choices=[0,1])



    parser.add_argument('--c_loss_type', type=str, default="l2")

    parser.add_argument('--split_val', type=float, default=0.2)
    parser.add_argument('--lambda_', type=float, default=1)
    parser.add_argument('--target_layer', type=str, default="fc2")
    parser.add_argument('--first_unfreeze_layer', type=str, default="block5_conv1")
    parser.add_argument('--network', type=str, default="vgg")
    parser.add_argument('--ref_filename', type=str, default="reference_train")
    parser.add_argument('--tar_train_filename', type=str, default="target_train")
    parser.add_argument('--tar_test_filename', type=str, default="target_test")
    parser.add_argument('--alien_filename', type=str, default="alien_test")

    return parser



if __name__ == "__main__":
    auc = []
    eer = []
    tp, fp, tn,fn = [],[],[],[]
    args = get_train_parser().parse_args()
    output_path = os.path.join(args.output_path, args.name)


    trainer = experiment_utils.Trainer(args)
    trainer.create_experiment_dir()
    trainer.set_ready()
    trainer.write_train_data()
    trainer.train()
    experiment = experiment_utils.Experiment(trainer.args.output_path, trainer.args.epochs, target_num=None, alien_num=None)
    eer_, auc_, tn_, fp_, fn_, tp_ = get_roc_curve(experiment.get_data_scores, "roc_results", experiment.model,
                  experiment.templates, experiment.aliens_positive, experiment.aliens_negative,
                  trainer.args.output_path)

    print(eer_, auc_, tn_, fp_, fn_, tp_)


