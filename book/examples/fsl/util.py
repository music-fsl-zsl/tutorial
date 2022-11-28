from typing import List, Dict, Tuple, Union

import random

import torch
import numpy as np
import librosa


def load_excerpt(audio_path: str, duration: float, sample_rate: int):
    """
    Load an excerpt of audio from a file.

    Returns a dictionary with the following keys:
        - audio: a torch.Tensor of shape (1, samples)
        - offset: the offset (in seconds) of the excerpt
        - duration: the duration (in seconds) of the excerpt
    """
    total_duration = librosa.get_duration(filename=audio_path)
    if total_duration < duration:
        raise ValueError(f"Audio file {audio_path} is too short"
               f"to extract an excerpt of duration {duration}")
    offset = random.uniform(0, total_duration - duration)
    audio, sr = librosa.load(audio_path, sr=sample_rate, 
                            offset=offset, duration=duration, 
                            mono=True)
    if audio.ndim == 1:
        audio = audio[None, :]
    return {
        "audio": torch.tensor(audio), 
        "offset": offset, 
        "duration": duration
    }


def collate_list_of_dicts(items: List[Dict]):
    """
    Collate a list of dictionaries into a single dictionary.
    """
    out = {}
    for d in items:
        for k, v in d.items():
            if k not in out:
                out[k] = []
            out[k].append(v)

    tensored = {}
    for k, v in out.items():
        if isinstance(v[0], torch.Tensor):
            tensored[k] = torch.stack(v)
        elif isinstance(v[0], str):
            tensored[k] = tuple(v)
        else:
            tensored[k] = v
    return tensored


def split(classes: List[str], split_percentages: Tuple[float]):
    """
    Split a list of classes into subsets, according to the given percentages.
    """
    assert sum(split_percentages) == 1.0, "Ratios must sum to 1"
    split_indices = [int(len(classes) * p) for p in split_percentages]
    splits = []
    start = 0
    for end in split_indices:
        splits.append(classes[start:start+end])
        start = end
    return splits

def dim_reduce(
        embeddings: List[np.ndarray], 
        color_labels: List[Union[int, str]], 
        marker_labels: List[int] = None,
        n_components: int = 3, 
        method: str= 'umap', 
        title: str = ''
    ):
    import plotly.express as px
    import umap
    import pandas as pd

    if method == 'umap':
        import umap
        reducer = umap.UMAP(n_components=n_components)
    elif method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=n_components)
    elif method == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=n_components)
    else:
        raise ValueError(f'dunno how to do {method}')
 
    proj = reducer.fit_transform(embeddings)

    if n_components == 2:
        df = pd.DataFrame(dict(
            x=proj[:, 0],
            y=proj[:, 1],
            label=color_labels
        ))
        fig = px.scatter(
            df, x='x', y='y', 
            color='label',
            title=title, 
            symbol=marker_labels
        )
    elif n_components == 3:
        df = pd.DataFrame(dict(
            x=proj[:, 0],
            y=proj[:, 1],
            z=proj[:, 2],
            label=color_labels
        ))
        fig = px.scatter_3d(
            df, x='x', y='y', z='z',
            color='label',
            symbol=marker_labels,
            title=title
        )
    else:
        raise ValueError("cant plot more than 3 components")

    fig.update_traces(marker=dict(size=6,
                                  line=dict(width=1,
                                            color='DarkSlateGrey')),
                      selector=dict(mode='markers'))

    return fig


def plotly_fig_to_tensor(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    import io
    import torchvision

    buf = io.BytesIO()
    figure.write_image(buf, format='png')
    buf.seek(0)

    image = torchvision.io.decode_png(
        torch.tensor(
            np.frombuffer(buf.getvalue(), dtype=np.uint8)
        )
    )

    return image
