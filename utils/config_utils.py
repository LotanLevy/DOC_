

import argparse
import json
import os


def write_config_file(parser_args):
    if not os.path.exists(parser_args.output_path):
        os.makedirs(parser_args.output_path)
    with open(os.path.join(parser_args.output_path, "train_settings.json"), 'wt') as json_file:
        json.dump(vars(parser_args), json_file)


def read_config_file(config_path, parser):
    with open(config_path, 'rt') as f:
        t_args = argparse.Namespace()
        json_items = json.load(f)
        args_list = []
        for arg_name, arg in json_items.items():
            args_list.append("--"+arg_name)
            args_list.append(str(arg))


        # t_args.__dict__.update(json.load(f))
        # args = parser.parse_args(namespace=t_args)
        args = parser.parse_args(args_list)

        return args

