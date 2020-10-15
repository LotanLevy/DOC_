
import os
from tensorflow.keras.models import Model
from dataloader import construct_with_files
from utils.config_utils import write_config_file, read_config_file
from utils.network_utils import get_network_functions
from dataloader import get_doc_loaders, parse_cls2label_map
from train import train, get_compactness_loss
from train_main import get_train_parser
import numpy as np
from test_utils.ROC_graph import get_scores_function, get_distance_func





class Experiment:
    def __init__(self, experiment_dir, weights_of_epoch,
                 templates_num=40, target_num=70, alien_num=70):
        self.trainer = Trainer.build_trainer(os.path.join(experiment_dir, "train_settings.json"), get_train_parser())
        self.trainer.set_ready()
        self.experiment_name = self.trainer.args.name
        self.experiment_dir = experiment_dir
        self.features_layer = self.trainer.args.target_layer
        self.model = self.build_model(experiment_dir, self.trainer.network_constractor, weights_of_epoch)
        self.get_data_scores = get_scores_function(self.trainer.args.c_loss_type)
        self.distance_func = get_scores_function(self.trainer.args.c_loss_type)
        np.random.seed(12345)

        self.templates, self.templates_paths = self.get_data_from_files(self.trainer.args.tar_train_filename, templates_num)
        self.target, self.target_paths = self.get_data_from_files(self.trainer.args.tar_test_filename, target_num)
        self.aliens, self.aliens_paths = self.get_data_from_files(self.trainer.args.alien_filename, alien_num, shuffle=True)

        self.aliens_positive, self.aliens_negative = self.split_positive_negative_aliens(self.trainer.args.alien_filename, alien_num, shuffle=True)



    def build_model(self, experiment_dir, network_constractor, epoch_num):
        model = network_constractor()
        ckpt_path = os.path.join(experiment_dir, "ckpts")
        if os.path.exists(ckpt_path):
            model.load_weights(os.path.join(ckpt_path, "weights_after_{}_epochs".format(epoch_num))).expect_partial()
            print(self.experiment_name + "'s ckpts loaded")
        else:
            print(self.experiment_name + " has not the required weights, loads imagenet weights only, instead")
        return Model(inputs=model.input, outputs=model.get_layer(self.features_layer).output)

    def get_data_from_files(self, filename, data_num, shuffle=False):
        paths, labels = os.path.join(self.experiment_dir, filename + "_paths.txt"), os.path.join(self.experiment_dir, filename + "_labels.txt")
        dataloader = construct_with_files(paths, labels, self.trainer.args.batchsize,
                                          (self.trainer.args.input_size, self.trainer.args.input_size),
                                          self.trainer.args.cls_num, False, self.trainer.preprocessing_func)
        return dataloader.get_all_data(size=data_num, shuffle=shuffle)

    def split_positive_negative_aliens(self, filename, data_num, shuffle=False):
        paths, labels = os.path.join(self.experiment_dir, filename + "_paths.txt"), os.path.join(self.experiment_dir,
                                                                                                 filename + "_labels.txt")
        dataloader = construct_with_files(paths, labels, self.trainer.args.batchsize,
                                          (self.trainer.args.input_size, self.trainer.args.input_size),
                                          self.trainer.args.cls_num, False, self.trainer.preprocessing_func)
        positive, _ = dataloader.get_all_data_by_label(1, size=data_num, shuffle=shuffle)
        negative, _ = dataloader.get_all_data_by_label(0, size=data_num, shuffle=shuffle)
        return positive, negative



class Trainer:
    def __init__(self, train_config_args):
        self.args = train_config_args

    def create_experiment_dir(self):
        output_dir = os.path.join(self.args.output_path, self.args.name)
        self.args.output_path = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        write_config_file(self.args)
        print("Experiment dir and settings file created")


    def set_ready(self):
        self.preprocessing_func, self.network_constractor = get_network_functions(self.args.network, self.args.cls_num, self.args.input_size)
        alien_cls2label = None if self.args.alien_cls2label is None else parse_cls2label_map(self.args.alien_cls2label)
        self.tar_train_loader, self.ref_loader, \
        self.tar_test_loader, self.alien_test_loader = get_doc_loaders(self.args.ref_dir, self.args.tar_dir,
                                                             self.args.alien_dir, self.args.batchsize,
                                                             (self.args.input_size, self.args.input_size),
                                                             self.args.split_val, self.args.cls_num, shuffle=True,
                                                             preprocess_func=self.preprocessing_func,
                                                                       alien_cls2label=alien_cls2label)
        self.compactness_loss = get_compactness_loss(self.args.c_loss_type, self.args.lambda_,self.args.batchsize)
        print("Network and dataloaders were created")

    def write_train_data(self):
        self.ref_loader.write_data(self.args.output_path, self.args.ref_filename)
        self.tar_train_loader.write_data(self.args.output_path, self.args.tar_train_filename)
        self.tar_test_loader.write_data(self.args.output_path, self.args.tar_test_filename)
        self.alien_test_loader.write_data(self.args.output_path, self.args.alien_filename)
        print("Data files created")


    def train(self):
        return train(self.tar_train_loader, self.ref_loader, self.args.epochs, self.args.first_unfreeze_layer,
                     self.compactness_loss, self.args.output_path,
                     self.network_constractor, self.args.batchsize, self.args.target_layer)

    @staticmethod
    def build_trainer(config_path, parser):
        configs = read_config_file(config_path, parser)
        return Trainer(configs)








