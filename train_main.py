

import argparse
import datetime

# from utils.experiment_utils import Trainer
from utils import experiment_utils

def get_train_parser():
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--ref_dir', type=str, required=True)
    parser.add_argument('--tar_dir', type=str, required=True)
    parser.add_argument('--alien_dir', type=str, required=True)
    parser.add_argument('--alien_cls2label', type=str, default=None)


    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--name', type=str, default=datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    parser.add_argument('--batchsize', '-bs', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--cls_num', type=int, default=1000)


    parser.add_argument('--c_loss_type', type=str, default="l2", choices=["l1", "l2"])

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
    trainer = experiment_utils.Trainer(get_train_parser().parse_args())
    trainer.create_experiment_dir()
    trainer.set_ready()
    trainer.write_train_data()
    trainer.train()

