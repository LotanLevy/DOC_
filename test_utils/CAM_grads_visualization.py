from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import imagenet_utils
import imutils
import re
import cv2
from tensorflow.keras.models import Model
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from test_utils.ROC_graph import get_distance_func


class GradCAM:
    def __init__(self, distance_func, model, templates):

        self.model = Model(inputs=model.input, outputs=model.layers[-1].output)
        self.templates = templates
        self.layerName = self.find_target_layer()
        self.distance_func = distance_func

    def find_target_layer(self):

        for layer in reversed(self.model.layers):
            # check to see if the layer has a 4D output
            if len(layer.output_shape) == 4:
                return layer.name
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")



    def compute_heatmap(self, image,eps=1e-8):

        gradModel = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output,
                     self.model.output])

        with tf.GradientTape() as tape:

            inputs = tf.cast(image, tf.float32)
            inputs = tf.Variable(inputs)
            tape.watch(inputs)

            (convOutputs, predictions) = gradModel(inputs)
            (t_convOutputs, t_predictions) = gradModel(self.templates)
            train = tf.reshape(t_predictions, (len(self.templates), -1))
            test = tf.reshape(predictions, (len(image), -1))
            # losses = tf.reduce_mean(tf.keras.backend.pow(tf.keras.backend.abs(test - train), 1), axis=-1)
            losses = self.distance_func(train, test)

            loss = tf.math.reduce_min(losses)

        grads = tape.gradient(loss, convOutputs)
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads

        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]

        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))

        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")
        # return the resulting heatmap to the calling function
        return heatmap, loss

    def overlay_heatmap(self, heatmap, image, alpha=0.5,
                        colormap=cv2.COLORMAP_JET ):

        heatmap = cv2.applyColorMap(heatmap, colormap)
        output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)

        return (heatmap, output)



def image_name(image_path):
  try:
    regex = ".*[\\/|\\\](.*)[\\/|\\\](.*).(jpg|JPEG)"
    m = re.match(regex, image_path)
    return m.group(1) + "_" + m.group(2)
  except:
    print(image_path)
    return None


def get_gradCam_image(experiments, image, image_path, output_path, loss_norm_dict):
    orig = np.array(Image.open(image_path).convert('RGB'))

    all_outputs = []
    labels_str = ""

    losses_dict = dict()


    for name, experiment in experiments.items():
      model = experiment.model
      cam = GradCAM(experiment.distance_func, model, experiment.templates)

      loss_ = experiment.get_data_scores(model, experiment.templates, image)
      losses_dict[name] = np.float(loss_)

      heatmap, loss = cam.compute_heatmap(np.copy(image))
      print(loss, loss_)

      heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
      (heatmap, output) = cam.overlay_heatmap(heatmap, orig, alpha=0.5)
      all_outputs.append(np.hstack([orig, heatmap, output]))

      loss_norm = loss_norm_dict[name]
      labels_str += "model {}: loss {}\n".format(name, loss/loss_norm)
    output = np.vstack(all_outputs)
    output = imutils.resize(output, height=2100)

    fig = plt.figure()
    plt.imshow(output)
    plt.title(labels_str)
    plt.savefig(os.path.join(output_path, image_name(image_path)), bbox_inches='tight')
    plt.close(fig)
    return losses_dict


def get_results_for_imagesdir(models_dict, images, paths, output_path, loss_norm_dict):

  models_losses = dict()
  for model_name in models_dict:
    models_losses[model_name] = []

  for i in range(len(paths)):
    image = np.expand_dims(images[i], axis=0)
    losses_dict = get_gradCam_image(models_dict, image, paths[i], output_path, loss_norm_dict)
    for model_name in models_dict:
      models_losses[model_name].append(losses_dict[model_name])
  return models_losses
