from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os


def get_data_scores(model, tempates, data_batch):
    templates_preds = model.predict(tempates)
    test_preds = model.predict(data_batch)

    templates_preds = templates_preds.reshape((len(tempates), -1))
    test_preds = test_preds.reshape((len(data_batch), -1))

    test_s_losses = np.zeros(len(test_preds))
    for i in range(len(test_preds)):
        losses = tf.keras.losses.mean_squared_error(templates_preds, tf.expand_dims(test_preds[i], axis=0))
        test_s_losses[i] = np.array(tf.math.reduce_min(losses))

    return test_s_losses


def get_roc_curve(model, templates, negative_data, positive_data, output_path):
    roc_path = os.path.join(output_path, "roc_graphs")

    if not os.path.exists(roc_path):
        os.makedirs(roc_path)
    # Abnormal score
    Z1 = get_data_scores(model, templates, negative_data)
    Z2 = get_data_scores(model, templates, positive_data)

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
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(True)
    plt.savefig(os.path.join(roc_path, "roc_graph.png"))
    plt.show()
