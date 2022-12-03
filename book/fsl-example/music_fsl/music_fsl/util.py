from typing import List, Dict, Tuple, Union
import tempfile
from pathlib import Path
import random

import torch
import numpy as np
import librosa
from IPython import display
import matplotlib.pyplot as plt

def batch_device(batch: dict, device: str = "cuda"):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    return batch

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
        if isinstance(v[0], np.ndarray):
            tensored[k] = torch.tensor(v)
        elif isinstance(v[0], torch.Tensor):
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
        embeddings: np.ndarray, 
        n_components: int = 3,
        method: str= 'umap',
    ):
    """
    Reduce the dimensionality of a set of embeddings.

    Args:
        embeddings: a numpy array of shape (n_samples, n_features)
        n_components: the number of components to reduce to
        method: the dimensionality reduction method to use. Must be one of
            ('umap', 'pca', 'tsne')

    Returns:
        a numpy array of shape (n_samples, n_components)
    """

    if method == 'umap':
        import umap
        reducer = umap.UMAP(
            n_neighbors=5, 
            n_components=n_components, 
            metric='euclidean'
        )
    elif method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(
            n_components=n_components, 
            init='pca', 
            learning_rate='auto'
        )
    
    elif method == 'pca':
        from sklearn.decomposition import PCA
        reducer =  PCA(n_components=n_components)
    else:
        raise ValueError(f'dunno how to do {method}')
 
    proj = reducer.fit_transform(embeddings)

    return proj


def embedding_plot(
        proj: np.ndarray, 
        color_labels: List[Union[int, str]], 
        marker_labels: List[int] = None,
        title: str = ''
    ):
    """
    Plot a set of embeddings that have been reduced using dim_reduce.

    Args:
        proj: a numpy array of shape (n_samples, n_components)
        color_labels: a list of labels to color the points by
        marker_labels: a list of labels to use as markers
        title: the title of the plot

    Returns:
        a plotly figure object
    """
    import plotly.express as px
    import pandas as pd
    
    n_components = proj.shape[-1]
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
        raise ValueError(f"can only plot 2 or 3 components but got {n_components}")

    fig.update_traces(marker=dict(size=6,
                                  line=dict(width=1,
                                            color='DarkSlateGrey')),
                      selector=dict(mode='markers'))

    return fig


def plotly_fig_to_tensor(figure, width=800, height=600):
    # Save the plot to a PNG in memory.
    import io
    import torchvision

    buf = io.BytesIO()
    figure.write_image(
        buf, format='png', 
        width=width, height=height
    )
    buf.seek(0)

    image = torchvision.io.decode_png(
        torch.tensor(
            np.frombuffer(buf.getvalue(), dtype=np.uint8)
        )
    )

    return image

def widget(audio_path, title=None):

    # compute the log Mel spectrogram of the audio
    audio, sr = librosa.load(audio_path, mono=True)
    log_mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=sr/2)
    log_mel_spectrogram = librosa.power_to_db(log_mel_spectrogram, ref=np.max)

    # create a temporary file for the log Mel spectrogram
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        # plot the log Mel spectrogram
        plt.figure(frameon=False)
        plt.imshow(log_mel_spectrogram, cmap='magma')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')

        # create the x-axis tick labels in Hz
        yticks = librosa.mel_frequencies(n_mels=128, fmax=sr/2)
        yticks = np.round(yticks).astype(int)
        yticks = np.flip(yticks)
        plt.yticks(np.arange(0, 128, 10), yticks[::10])

        title = title or Path(audio_path).stem
        plt.title(title)

        # save the plot to the temporary file
        plt.savefig(f.name)
        plt.clf()

        # create the image widget from the temporary file
        image_widget = display.Image(
            filename=f.name, 
            embed=True
        )

    # create the audio widget from the audio file
    audio_widget = display.Audio(
        filename=audio_path, 
        autoplay=False, 
        embed=True
    )


    return display.display(image_widget), display.display(audio_widget)