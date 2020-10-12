import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
import re


def read_paths_file(path_to_paths_file):
    with open(path_to_paths_file, 'r') as f:
        return [line[:-1] for line in f]


def read_labels_file(path_to_labels_file):
    with open(path_to_labels_file, 'r') as f:
        return [int(line[:-1]) for line in f]


"""
Data iterator that loads the data from a path only when a batch called (next and get_all_data)
"""


class DirIter:
    def __init__(self, paths, labels, batch_size, input_size, classes_num, shuffle=False, preprocess_func=lambda x: x):
        if labels is not None:
            assert (len(paths) == len(labels))
            self.labels = np.array(labels)
        else:
            self.labels = None
        self.preprocess_func = preprocess_func

        self.paths = paths
        self.classes_num = classes_num
        self.batch_size = batch_size
        self.indices = np.arange(len(self.paths)).astype(np.int)
        self.input_size = input_size
        self.shuffle = shuffle
        self.on_epoch_end()
        self.cls2label = None

    def __len__(self):
        return len(self.paths)

    def on_epoch_end(self):
        if self.shuffle and len(self.indices) > 0:
            np.random.shuffle(self.indices)
        self.cur_idx = 0

    @staticmethod
    def image_cls_and_name(image_path):
        try:
            regex = ".*[\\/|\\\](.*)[\\/|\\\](.*).(jpg|JPEG)"
            m = re.match(regex, image_path)
            return m.group(1), m.group(2)
        except:
            print("can't parse image's name as iterator format (cls/name)" + image_path)
            return None, None

    def load_img(self, image_path):
        image = Image.open(image_path, 'r')
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize(self.input_size, Image.NEAREST)
        image = np.array(image).astype(np.float32)
        return np.expand_dims(image, axis=0)

    def next(self):
        relevant_indices = self.indices[self.cur_idx: self.cur_idx + self.batch_size]
        self.cur_idx += self.batch_size
        images = []
        images = np.concatenate([self.load_img(self.paths[i]) for i in relevant_indices])
        if self.labels is not None:
            labels = self.labels[relevant_indices]
            labels = tf.keras.utils.to_categorical(labels, num_classes=self.classes_num)
        else:
            labels = None
        return self.preprocess_func(images), labels

    def get_all_data(self, size=None):
        if size is None:
            size = len(self.paths)
        relevant_paths = [self.paths[i] for i in self.indices[:size]]
        images = np.concatenate([self.load_img(path) for path in relevant_paths])
        return self.preprocess_func(images), relevant_paths

    def has_next(self):
        return self.cur_idx + self.batch_size < len(self.indices)

    def set_cls2label_map(self, map):
        self.cls2label = map

    def write_data(self, output_path, loader_name):
        with open(os.path.join(output_path, loader_name + "_paths.txt"), 'w') as f:
            for p in self.paths:
                f.write(p + "\n")
        with open(os.path.join(output_path, loader_name + "_labels.txt"), 'w') as f:
            for l in self.labels:
                f.write(str(l) + "\n")


def construct_with_files(path_file, label_file, batch_size, input_size, classes_num, shuffle=False,
                         preprocess_func=lambda x: x):
    paths = read_paths_file(path_file)
    if label_file is not None:
        labels = read_labels_file(label_file)
    else:
        labels = None
    return DirIter(paths, labels, batch_size, input_size, classes_num, shuffle, preprocess_func)


"""
Data loader that loads data from a dir of sub dirs, each sub dir contains data of one class.
Works similarly to directory iterator, but it loads the data from a path only when a batch called (next and get_all_data)
"""


def get_iterators_by_root_dir(root_dir, batch_size, input_size, split_val, classes_num, shuffle=False,
                              preprocess_func=lambda x: x):
    dirs = os.listdir(root_dir)
    length = len(max(dirs, key=len))

    for dir in dirs:  # Handle the sort problem pads the clas num with '0'
        if len(dir) < length:
            zeros = "0" * (length - len(dir))
            new_name = zeros + dir

            os.rename(os.path.join(root_dir, dir), os.path.join(root_dir, new_name))
            print("old {}, new {}".format(dir, new_name))

    paths = []
    labels = []
    cls2label = dict()
    label_idx = 0
    for sub_dir in sorted(os.listdir(root_dir)):

        full_path = os.path.join(root_dir, sub_dir)
        if not os.path.isdir(full_path):
            continue
        cls2label[sub_dir] = label_idx
        for file in os.listdir(full_path):
            paths.append(os.path.join(full_path, file))
            labels.append(label_idx)
        label_idx += 1

    print(cls2label)

    assert len(paths) == len(labels)
    if len(cls2label) != classes_num:
        print("classes in directory doesn't match classes_num")

    if split_val > 0:
        X_train, X_test, y_train, y_test = train_test_split(paths, labels, test_size=split_val, shuffle=shuffle)
    else:
        X_train, X_test, y_train, y_test = paths, [], labels, []

    train_iter = DirIter(X_train, y_train, batch_size, input_size, classes_num, shuffle=True,
                         preprocess_func=preprocess_func)
    val_iter = DirIter(X_test, y_test, batch_size, input_size, classes_num, shuffle=True,
                       preprocess_func=preprocess_func)

    train_iter.set_cls2label_map(cls2label)
    val_iter.set_cls2label_map(cls2label)
    return train_iter, val_iter


def get_doc_loaders(ref_path, tar_path, alien_path, batchsize, input_size, split_val, cls_num, shuffle=False,
                    preprocess_func=lambda x: x):
    ref_loader, i1 = get_iterators_by_root_dir(ref_path, batchsize, input_size, 0, cls_num, shuffle=shuffle,
                                               preprocess_func=preprocess_func)
    train_s_loader, test_s_loader = get_iterators_by_root_dir(tar_path, batchsize, input_size, split_val, cls_num,
                                                              shuffle=shuffle, preprocess_func=preprocess_func)
    test_alien_loader, i2 = get_iterators_by_root_dir(alien_path, batchsize, input_size, 0, cls_num, shuffle=shuffle,
                                                      preprocess_func=preprocess_func)
    print(len(ref_loader), len(i1))
    print(len(train_s_loader), len(test_s_loader))
    print(len(test_alien_loader), len(i2))

    return train_s_loader, ref_loader, test_s_loader, test_alien_loader
