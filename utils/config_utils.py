

import argparse
import json
import os


def write_config_file(parser_args):
    if not os.path.exists(parser_args.output_path):
        os.makedirs(parser_args.output_path)
    with open(os.path.join(parser_args.output_path, "train_settings.json"), 'w') as json_file:
        json.dump(vars(parser_args), json_file)


def read_config_file(config_path, parser):
    with open(config_path, 'rt') as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=t_args)
        return args

