from tensorflow.keras.losses import Loss
import re
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K


class MeanPError(Loss):

    def __init__(self, power):
        Loss.__init__(self)
        self.power = power

    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        return tf.reduce_mean(tf.math.pow(tf.math.abs(y_pred - y_true), self.power), axis=-1)



def get_distance_func(c_loss_type):
    m = re.match("l(.*)", c_loss_type)
    if m is not None:
        return MeanPError(str(m.group(1)))
    # return m.group(1)
    # c_loss_func = None
    # if c_loss_type == "l2":
    #     # return MeanPError(2)
    #     c_loss_func = tf.keras.losses.mean_squared_error
    # elif c_loss_type == "l1":
    #     c_loss_func = tf.keras.losses.mean_absolute_error
    # return c_loss_func


def get_scores_function(distance_func):

    if distance_func is not None:
        def get_data_scores(model, tempates, data_batch):
            templates_preds = model.predict(tempates)
            test_preds = model.predict(data_batch)
            templates_preds = templates_preds.reshape((len(tempates), -1))
            test_preds = test_preds.reshape((len(data_batch), -1))
            test_s_losses = np.zeros(len(test_preds))
            for i in range(len(test_preds)):
                losses = distance_func(templates_preds, tf.expand_dims(test_preds[i], axis=0))
                test_s_losses[i] = np.array(tf.math.reduce_min(losses))
            return test_s_losses
        return get_data_scores


class LossHelper:
    def __init__(self, compact_loss_type, loss_weight, regularization):
        m = re.match("l(.*)", compact_loss_type)
        if m is not None:
            self.p = int(m.group(1))
        else:
            print("The given compact loss doesn't support")
        self.distance_func = MeanPError(self.p)
        self.compact_loss_weight = loss_weight
        self.compact_regularization = regularization

    def get_compact_loss(self, n_dim):# n_dim - number of features vecs
        def compactness_loss(y_true, y_pred):
            k_dim = np.shape(y_pred)[1]  # feature vec dim
            beta = (1 / k_dim) * (n_dim / (n_dim - 1)) ** self.p
            sigma = K.sum(K.abs((y_pred - K.mean(y_pred, axis=0))) ** self.p, axis=[1])# sum the features space
            lc = beta * sigma

            # lc = 1 / (k_dim * n_dim) * n_dim ** self.p * K.sum(K.abs((y_pred - K.mean(y_pred, axis=0))) ** self.p, axis=[1]) / (
            #             (n_dim - 1) ** self.p)
            return lc * self.compact_loss_weight
        print("l{} compactness loss initialized".format(self.p))
        return compactness_loss

    def get_scores_function(self):

        if self.distance_func is not None:
            def get_data_scores(model, tempates, data_batch):
                templates_preds = model.predict(tempates)
                test_preds = model.predict(data_batch)
                templates_preds = templates_preds.reshape((len(tempates), -1))
                test_preds = test_preds.reshape((len(data_batch), -1))
                test_s_losses = np.zeros(len(test_preds))
                for i in range(len(test_preds)):
                    losses = self.distance_func(templates_preds, tf.expand_dims(test_preds[i], axis=0))
                    test_s_losses[i] = np.array(tf.math.reduce_min(losses))
                return test_s_losses

            return get_data_scores