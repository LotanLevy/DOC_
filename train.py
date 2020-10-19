import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import datetime
import numpy as np


# Loss function
def get_compactness_loss(type, lambda_, n_dim):
    if type == "l2":
        def compactness_loss2(y_true, y_pred):
            # n_dim = np.shape(y_pred)[0]  # number of features vecs
            k_dim = np.shape(y_pred)[1]  # feature vec dim
            lc = 1 / (k_dim * n_dim) * n_dim ** 2 * K.sum((y_pred - K.mean(y_pred, axis=0)) ** 2, axis=[1]) / (
                        (n_dim - 1) ** 2)
            return lc * lambda_
        print("l2 compactness loss initialized")
        return compactness_loss2
    if type == "l1":
        def compactness_loss1(y_true, y_pred):
            # n_dim = np.shape(y_pred)[0]  # number of features vecs
            k_dim = np.shape(y_pred)[1]  # feature vec dim
            lc = 1 / (k_dim * n_dim) * n_dim * K.sum(K.abs(y_pred - K.mean(y_pred, axis=0)) , axis=[1]) / (
                        (n_dim - 1))
            return lc * lambda_
        print("l1 compactness loss initialized")
        return compactness_loss1



# Learning
def train(target_dataloader, reference_dataloader, epoch_num, first_trained_layer_name, compactness_loss, output_dir,
          network_constractor, batchsize, target_layer_name, not_use_d_lose=False):
    # output dirs !
    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    epochs_log_dir = os.path.join(os.path.join(output_dir, "epochs_logs/" + time))
    ckpt_path = os.path.join(output_dir, "ckpts/")
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    print("Model build...")
    network = network_constractor()
    # Freeze weight
    for layer in network.layers:
        if layer.name == first_trained_layer_name:
            break
        else:
            layer.trainable = False
    model_t = Model(inputs=network.input, outputs=network.get_layer(target_layer_name).output)
    model_r = Model(inputs=model_t.input, outputs=network.layers[-1].output)

    # Compile
    train_accuracy = tf.keras.metrics.CategoricalAccuracy()
    optimizer = SGD(lr=5e-5, decay=0.00005)
    model_r.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=[train_accuracy])
    model_t.compile(optimizer=optimizer, loss=compactness_loss)

    # Prints run settings
    model_t.summary()
    model_r.summary()
    print("x_target is", len(target_dataloader), 'samples')
    print("x_ref is", len(reference_dataloader), 'samples')

    # run loggers
    epochs_writer = tf.summary.create_file_writer(logdir=epochs_log_dir)
    outputs = {"Descriptive_loss": [], "Compact_loss": [], "Accuracy": []}

    print("training...")
    # Learning
    for epochnumber in range(epoch_num):
        lc, ld, accuracy = [], [], []  # epoch loaders
        total = int(len(target_dataloader) / batchsize)
        with tqdm(total=total, position=0, leave=True) as pbar:
            for i in tqdm(range(total), position=0, leave=True):
                pbar.update()
                # Load data for batch size
                batch_is_ready = False
                while not batch_is_ready:
                    try:
                        batch_target, _ = target_dataloader.next()
                        batch_is_ready = True
                    except Exception as e:  # some error in loading the data
                        if not target_dataloader.has_next():
                            target_dataloader.on_epoch_end()
                        print(e)
                        continue

                # target data
                # Get loss while learning
                lc.append(model_t.train_on_batch(batch_target, np.zeros((batchsize, 4096))))

                if not not_use_d_lose:
                    batch_ref, batch_y = reference_dataloader.next()

                    # reference data
                    # Get loss while learning
                    ref_output = model_r.train_on_batch(batch_ref, batch_y)
                    ld.append(ref_output[0])
                    accuracy.append(ref_output[1])

        target_dataloader.on_epoch_end()
        reference_dataloader.on_epoch_end()

        outputs["Descriptive_loss"].append(np.mean(ld))
        outputs["Compact_loss"].append(np.mean(lc))
        outputs["Accuracy"].append(np.mean(accuracy))
        with epochs_writer.as_default():
            for key in outputs:
                tf.summary.scalar(key, outputs[key][-1], step=epochnumber + 1)
        print("epoch: {}: d loss {}, c loss {}, accuracy {}".format(epochnumber + 1, outputs["Descriptive_loss"][-1],
                                                                    outputs["Compact_loss"][-1],
                                                                    outputs["Accuracy"][-1]))

        checkpoint_path = "weights_after_{}_epochs".format(epochnumber + 1)
        network.save_weights(os.path.join(ckpt_path, checkpoint_path))

    for item_name in outputs:
        # Result graph
        plt.plot(outputs[item_name], label=item_name)
        plt.xlabel("epoch")
        plt.legend()
        plt.savefig(os.path.join(output_dir, item_name + "_graph.jpg"))
        plt.show()

    return model_t
