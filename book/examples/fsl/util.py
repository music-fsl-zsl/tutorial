from typing import List, Dict, Tuple

import random

import torch
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