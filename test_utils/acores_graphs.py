from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import math
from skimage.transform import resize
import matplotlib.pyplot as plt
import os


def map_path_to_class(paths):
    paths2cls = dict()
    for path in paths:
        cls = path.split('\\')[-2]
        paths2cls[path] = cls
    return paths2cls


def getImage(path, zoom):
    image = plt.imread(path)
    image = resize(image, (224, 224))
    return OffsetImage(image, zoom=zoom)


def create_images_graph(output_path, paths, scores, name, zoom, columns, max_objects=None):
    scores_graph_path = os.path.join(output_path, "score_graphs")
    if not os.path.exists(scores_graph_path):
        os.makedirs(scores_graph_path)
    if max_objects is None:
        max_objects = len(scores)
    paths2cls = map_path_to_class(paths)
    indices = np.argsort(scores)[:max_objects]
    scores = scores[indices]

    step = 10

    x = list(range(columns)) * math.ceil(len(indices) / float(columns))

    x = [step * i for i in x]
    x = x[:len(scores)]
    fig, ax = plt.subplots()
    # ax.scatter(x, scores[indices])
    for i in range(max_objects):
        idx = indices[i]
        ab = AnnotationBbox(getImage(paths[idx], zoom), (x[i], scores[i]), frameon=False)
        ax.scatter(x[i], scores[i])
        ax.add_artist(ab)
    ax.update_datalim(np.column_stack([x, scores]))
    ax.autoscale(-1 * max(scores), max(scores) * 1.1)
    ax.set_xlim(-1, max(x) * 1.1)
    plt.ylabel("classifier score")
    plt.xlabel("axis without meaning")
    plt.title(name)
    plt.savefig(os.path.join(scores_graph_path, "scores visualization_of_{}.png".format(name)), dpi=500)
    plt.show()
