
import os
from tensorflow.keras.models import Model
from dataloader import construct_with_files



class Experiment:
    def __init__(self, experiment_name, experiment_dir, network_constractor, weights_of_epoch, preprocessing_func, features_idx=-2,
                 templates_num=40, target_num=70, alien_num=70, image_size=224):
        self.experiment_name = experiment_name
        self.experiment_dir = experiment_dir
        self.features_idx = features_idx
        self.model = self.build_model(experiment_dir, network_constractor, weights_of_epoch)
        self.templates, self.templates_paths = self.get_data_from_files(experiment_dir, "target_train_paths.txt", "target_train_labels.txt",
                                                  preprocessing_func, image_size, templates_num)# tuple of images, paths
        self.target, self.target_paths = self.get_data_from_files(experiment_dir, "target_test_paths.txt", "target_test_labels.txt",
                                                  preprocessing_func, image_size, target_num)# tuple of images, paths
        self.aliens, self.aliens_paths = self.get_data_from_files(experiment_dir, "alien_test_paths.txt", "alien_test_labels.txt",
                                                  preprocessing_func, image_size, alien_num)# tuple of images, paths


    def build_model(self, experiment_dir, network_constractor, epoch_num):
        model = network_constractor()
        ckpt_path = os.path.join(experiment_dir, "ckpts")
        if os.path.exists(ckpt_path):
            model.load_weights(os.path.join(ckpt_path, "weights_after_{}_epochs".format(epoch_num))).expect_partial()
        print(self.experiment_name + "'s ckpts loaded")
        return Model(inputs=model.input, outputs=model.layers[self.features_idx].output)

    def get_data_from_files(self, experiment_dir, paths_filename, labels_filename, preprocessing_func, image_size, data_num):
        paths, labels = os.path.join(experiment_dir, paths_filename), os.path.join(experiment_dir, labels_filename)
        dataloader = construct_with_files(paths, labels, 2, (image_size, image_size), 1000, False, preprocessing_func)
        return dataloader.get_all_data(size=data_num)







