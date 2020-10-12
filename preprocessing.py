from utils.augmentation_utils import Augmentor
from dataloader import get_iterators_by_root_dir
import argparse
import re
import os

def get_args():
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--input_size', type=int, nargs=2, default=(224, 224))
    return parser.parse_args()


to_aug_name_regex = ".*\\\((.(?!\\\))*)$"

if __name__=="__main__":
    args = get_args()
    print(args.images_dir)
    m = re.match(to_aug_name_regex, args.images_dir)
    to_augment_dir_name = m.group(1)
    print("starting augment dir " + to_augment_dir_name)
    augments_dir = os.path.join(args.output_path, to_augment_dir_name + "_augmented")
    if not os.path.exists(augments_dir):
        os.makedirs(augments_dir)
    iterator_dir, _ = get_iterators_by_root_dir(args.images_dir, 0, args.input_size, 0, 0)
    augmentor = Augmentor(horizontal_flip=True, vertical_flip=True, random_rotate=True, copy=True)
    augmentor.augment(iterator_dir, augments_dir)
