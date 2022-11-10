from typing import List, Dict

import torch
import librosa
import numpy as np
import IPython.display as ipd

def widget(path: str):
    return ipd.Audio(path, autoplay=False)

def load_excerpt(audio_path: str, duration: float, sample_rate: int):
    """
    Load an excerpt of audio from a file.
    """
    audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    if audio.ndim == 1:
        audio = audio[None, :]
    offset = np.random.randint(0, audio.shape[-1] - int(duration * sr))
    return {
        "audio": torch.Tensor(audio[:, offset:offset+int(duration*sr)]), 
        "offset": offset, 
        "duration": duration
    }

def collate_list_of_dicts(batch: List[Dict]):
    """
    Collate a list of dictionaries into a single dictionary.
    """
    out = {}
    for d in batch:
        for k, v in d.items():
            if k not in out:
                out[k] = []
            out[k].append(v)

    for k, v in out.items():
        if isinstance(v[0], torch.Tensor):
            print(k)
            out[k] = torch.stack(v)
        else:
            out[k] = v
    print(out)
    return out