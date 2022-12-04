import matplotlib.pyplot as plt
from matplotlib import cm
from IPython.display import Audio, display
from PIL import Image

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import numpy as np
import pandas as pd


def nearest_neighbor_clf_k(query_emb, class_to_mean_emb, class_txt_list, k=3):
    class_embs = []
    for i in range(len(class_txt_list)):
        class_embs.append(class_to_mean_emb[i])
    class_embs = np.stack(class_embs, 0)
    sim_matrix = cosine_similarity(query_emb, class_embs)
    df_sim = pd.DataFrame(sim_matrix).T
    top_idx = df_sim[0].sort_values(ascending=False).head(k)
    return top_idx.index, [class_txt_list[top_idx.index[i]] for i in range(k)]

def draw_pca_with_class(_embs, _labels, _class_txt_list):
    pca = PCA(n_components=2)
    _pca_proj = pca.fit_transform(_embs)
    cmap = cm.get_cmap('tab20')
    fig, ax = plt.subplots(figsize=(8,8))
    for _i in range(len(_class_txt_list)):
        indices = [_j for _j in range(len(_labels)) if _labels[_j] == _i]
        ax.scatter(_pca_proj[indices, 0],
                   _pca_proj[indices, 1], 
                   c=np.array(cmap(_i)).reshape(1,4), 
                   label=_class_txt_list[_i],
                   s=12.0,
                   alpha=1.0
                  )
        ax.annotate(_class_txt_list[_i], (_pca_proj[indices, 0], _pca_proj[indices, 1]))
    plt.show()

def draw_pca(_embs, _labels, _class_txt_list):
    pca = PCA(n_components=2)
    _pca_proj = pca.fit_transform(_embs)
    cmap = cm.get_cmap('tab20')
    fig, ax = plt.subplots(figsize=(8,8))
    for _i in range(len(_class_txt_list)):
        indices = [_j for _j in range(len(_labels)) if _labels[_j] == _i]
        ax.scatter(_pca_proj[indices, 0],
                   _pca_proj[indices, 1], 
                   c=np.array(cmap(_i)).reshape(1,4), 
                   label=_class_txt_list[_i],
                   s=12.0,
                   alpha=1.0
                  )
    ax.legend(fontsize='large', markerscale=3)
    plt.show()
    
def draw_tsne_with_class(_embs, _labels, _class_txt_list, dist='cosine'):
    tsne = TSNE(2, perplexity=10, n_iter=10000, verbose=0, metric=dist)
    _tsne_proj = tsne.fit_transform(_embs)
    cmap = cm.get_cmap('tab20')
    fig, ax = plt.subplots(figsize=(8,8))
    for _i in range(len(_class_txt_list)):
        indices = [_j for _j in range(len(_labels)) if _labels[_j] == _i]
        ax.scatter(_tsne_proj[indices, 0],
                   _tsne_proj[indices, 1], 
                   c=np.array(cmap(_i)).reshape(1,4), 
                   label=_class_txt_list[_i],
                   s=12.0,
                   alpha=1.0
                  )
        ax.annotate(_class_txt_list[_i], (_tsne_proj[indices, 0], _tsne_proj[indices, 1]))
    plt.show()

def draw_tsne(_embs, _labels, _class_txt_list, dist='cosine'):
    tsne = TSNE(2, perplexity=10, n_iter=10000, verbose=0, metric=dist)
    _tsne_proj = tsne.fit_transform(_embs)
    cmap = cm.get_cmap('tab20')
    fig, ax = plt.subplots(figsize=(8,8))
    for _i in range(len(_class_txt_list)):
        indices = [_j for _j in range(len(_labels)) if _labels[_j] == _i]
        ax.scatter(_tsne_proj[indices, 0],
                   _tsne_proj[indices, 1], 
                   c=np.array(cmap(_i)).reshape(1,4), 
                   label=_class_txt_list[_i],
                   s=12.0,
                   alpha=1.0
                  )
    ax.legend(fontsize='large', markerscale=3)
    plt.show()
    