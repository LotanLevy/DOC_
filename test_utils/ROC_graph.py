from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os


def get_distance_func(c_loss_type):
    c_loss_func = None
    if c_loss_type == "l2":
        c_loss_func = tf.keras.losses.mean_squared_error
    elif c_loss_type == "l1":
        c_loss_func = tf.keras.losses.mean_absolute_error
    return c_loss_func


def get_scores_function(c_loss_type):
    c_loss_func = get_distance_func(c_loss_type)

    if c_loss_func is not None:
        def get_data_scores(model, tempates, data_batch):
            templates_preds = model.predict(tempates)
            test_preds = model.predict(data_batch)
            templates_preds = templates_preds.reshape((len(tempates), -1))
            test_preds = test_preds.reshape((len(data_batch), -1))
            test_s_losses = np.zeros(len(test_preds))
            for i in range(len(test_preds)):
                losses = c_loss_func(templates_preds, tf.expand_dims(test_preds[i], axis=0))
                test_s_losses[i] = np.array(tf.math.reduce_min(losses))
            return test_s_losses
        return get_data_scores


def get_roc_curve(get_data_scores_func, title, model, templates, negative_data, positive_data, output_path):
    roc_path = os.path.join(output_path, "roc_graphs")

    if not os.path.exists(roc_path):
        os.makedirs(roc_path)
    # Abnormal score
    Z1 = get_data_scores_func(model, templates, negative_data)
    Z2 = get_data_scores_func(model, templates, positive_data)

    # Drawing of ROC curve
    y_true = np.zeros(len(negative_data) + len(positive_data))
    y_true[len(negative_data):] = 1  # 0:Normal, 1：Abnormal

    # Calculate FPR, TPR(, Threshould)
    fpr, tpr, _ = metrics.roc_curve(y_true, np.hstack((Z1, Z2)))

    # AUC
    auc = metrics.auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label='DeepOneClassification(AUC = %.2f)' % auc)
    plt.legend()
    plt.title(title + 'ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(True)
    plt.savefig(os.path.join(roc_path, title + "roc_graph.png"))
    plt.show()
