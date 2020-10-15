
import numpy as np
from PIL import Image
import os
from scipy.ndimage import rotate


def load_img(image_path, input_size):
    image = Image.open(image_path, 'r')
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(input_size, Image.NEAREST)
    image = np.array(image).astype(np.float32)
    return np.expand_dims(image, axis=0)


class Augmentor:

    AUGMENT_FUNCS = {"horizontal_flip" : np.fliplr,
                     "vertical_flip": np.flipud,
                     "random_rotate": lambda image: rotate(image, np.random.randint(0, 360)),
                     "copy": lambda image: image}

    def __init__(self, horizontal_flip=False, vertical_flip=False, random_rotate=False, copy=False):
        self.relevant_augs = dict()
        if horizontal_flip:
            self.relevant_augs["horizontal_flip"] = self.AUGMENT_FUNCS["horizontal_flip"]
        if vertical_flip:
            self.relevant_augs["vertical_flip"] = self.AUGMENT_FUNCS["vertical_flip"]
        if random_rotate:
            self.relevant_augs["random_rotate"] = self.AUGMENT_FUNCS["random_rotate"]
        if copy:
            self.relevant_augs["copy"] = self.AUGMENT_FUNCS["copy"]




    def aug_image(self, image, aug_func):
        image = image.astype(np.float)
        result = aug_func(image)
        return result




    def augment(self, directory_iterator, output_path):
        paths = directory_iterator.paths
        for path in paths:
            image = directory_iterator.load_img(path)[0]
            cls, name = directory_iterator.image_cls_and_name(path)
            cls_path = os.path.join(output_path, cls)
            if not os.path.exists(cls_path):
                os.makedirs(cls_path)
            for aug_name in self.relevant_augs:
                aug_image = np.uint8(self.aug_image(image, self.relevant_augs[aug_name]))
                Image.fromarray(aug_image).save(os.path.join(cls_path, name.replace(" " , "_") + "_" + aug_name+".jpg"))



