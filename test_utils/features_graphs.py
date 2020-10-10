from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os


def get_features_graph(model, templates, target_data, alien_data, output_path):
  templates_preds = model.predict(templates)
  target_preds = model.predict(target_data)
  alien_preds = model.predict(alien_data)

  templates_embedded = TSNE(n_components=2, random_state=123).fit_transform(templates_preds)
  target_embedded = TSNE(n_components=2, random_state=123).fit_transform(target_preds)
  alien_embedded = TSNE(n_components=2, random_state=123).fit_transform(alien_preds)

  plt.figure()
  plt.scatter(templates_embedded[:, 0],templates_embedded[:,1], label="Templates")
  plt.scatter(target_embedded[:, 0],target_embedded[:,1], label="Target")
  plt.scatter(alien_embedded[:, 0],alien_embedded[:,1], label="Alien")
  plt.title("features graph (TSNE)")
  plt.savefig(os.path.join(output_path, "features_graphs.png"))
  plt.legend()
  plt.grid(True)
  plt.show()
