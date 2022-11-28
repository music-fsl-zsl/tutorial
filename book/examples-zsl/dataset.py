import random
import os
from PIL import Image
from collections import defaultdict

import torch
import torch.nn.functional as F
import torchaudio
from torchaudio.transforms import Resample

SAMPLING_RATE = 16000
WAV_INPUT_LENGTH = 16000 * 3  
MEL_LENGTH = 63 * 3

class WordAudioDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        audio_path_list,
        audio_label_list,
        audio_transform,
        curr_word_classes,
        word_emb_dict,
        audio_dir = './data/TinySOL/audio'
    ) -> None:
        
        self.audio_paths = audio_path_list
        self.audio_dir = audio_dir
        self.audio_labels = audio_label_list
        self.audio_label_set = set(self.audio_labels)
        self.word_classes = curr_word_classes

        self.label_idx_to_word_emb_dict = {}
        for idx in range(len(curr_word_classes)):
            self.label_idx_to_word_emb_dict[idx] = word_emb_dict[curr_word_classes[idx]]
        
        self.audio_transform = audio_transform
        self.SAMPLING_RATE = SAMPLING_RATE
        self.WAV_INPUT_LENGTH = WAV_INPUT_LENGTH
        self.MEL_LENGTH = MEL_LENGTH
        
    def __getitem__(self, idx: int):
        _path = self.audio_paths[idx]
        wav, sr = self._load_item(os.path.join(self.audio_dir, _path))
        wav = self._resample_item(wav, sr)
        wav = self._get_random_wav_seg(wav)
        audio = self.audio_transform(wav)
        audio = audio.unsqueeze(0)
        
        pos_word_emb, neg_word_emb = self._get_pos_neg_inp(idx)

        return (
            audio, 
            pos_word_emb, 
            neg_word_emb, 
            self.audio_labels[idx], 
            self.word_classes[self.audio_labels[idx]]
        )
    
    def _load_item(self, path):
        waveform, sample_rate = torchaudio.load(path)
        return waveform[0], sample_rate

    def _resample_item(self, wav, org_sr):
        _resample = Resample(
            orig_freq=org_sr,
            new_freq=self.SAMPLING_RATE,
        )
        return _resample(wav)

    def _get_random_wav_seg(self, wav):
        if wav.shape[0] > self.WAV_INPUT_LENGTH:
            _start = random.randint(0, wav.shape[0] - self.WAV_INPUT_LENGTH)
            wav = wav[_start:_start + self.WAV_INPUT_LENGTH]
        else:
            wav = F.pad(wav, pad=(0, self.WAV_INPUT_LENGTH - wav.shape[0]), mode='constant', value=0)
        return wav

    def _get_sequential_wav_seg_list(self, wav):
        if wav.shape[0] > self.WAV_INPUT_LENGTH:
            num_segs = wav.shape[0] // self.WAV_INPUT_LENGTH
            wavs = [wav[i*self.WAV_INPUT_LENGTH:(i+1)*self.WAV_INPUT_LENGTH] for i in range(num_segs)]
        else:
            wav = F.pad(wav, pad=(0, self.WAV_INPUT_LENGTH - wav.shape[0]), mode='constant', value=0)
            wavs = [wav]
        return wavs

    def _get_pos_neg_inp(self, idx):
        pos_label = self.audio_labels[idx]
        neg_label = random.choice([x for x in list(self.audio_label_set) if x != pos_label])
        pos_word_emb = self.label_idx_to_word_emb_dict[pos_label]
        neg_word_emb = self.label_idx_to_word_emb_dict[neg_label]
        return torch.from_numpy(pos_word_emb), torch.from_numpy(neg_word_emb)

    def __len__(self) -> int:
        return len(self.audio_paths)


class ImageAudioDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        audio_path_list,
        audio_label_list,
        img_path_list,
        img_label_list,
        img_class_list,
        audio_transform,
        img_transform,
        audio_dir = './data/TinySOL/audio'
    ) -> None:
        
        self.audio_paths = audio_path_list
        self.audio_dir = audio_dir
        self.audio_labels = audio_label_list
        self.audio_label_set = set(self.audio_labels)
        self.img_class_list = img_class_list
        self.img_path_list = img_path_list
        self.img_label_list = img_label_list
        
        self.label_idx_to_img_paths_dict = defaultdict(list)
        for i in range(len(img_path_list)):
            self.label_idx_to_img_paths_dict[img_label_list[i]].append(img_path_list[i])
        
        self.audio_transform = audio_transform
        self.img_transform = img_transform
        self.SAMPLING_RATE = SAMPLING_RATE
        self.WAV_INPUT_LENGTH = WAV_INPUT_LENGTH
        self.MEL_LENGTH = MEL_LENGTH
        
    def __getitem__(self, idx: int):

        _path = self.audio_paths[idx]
        wav, sr = self._load_item(os.path.join(self.audio_dir, _path))
        wav = self._resample_item(wav, sr)
        wav = self._get_random_wav_seg(wav)
        audio = self.audio_transform(wav)
        audio = audio.unsqueeze(0)
        
        pos_label = self.audio_labels[idx]
        pos_img = random.sample(self.label_idx_to_img_paths_dict[pos_label], 1)[0]
        neg_label = random.choice([x for x in list(self.audio_label_set) if x != pos_label])
        neg_img = random.sample(self.label_idx_to_img_paths_dict[neg_label], 1)[0]
        
        pos_img_tensor = self.img_transform(Image.open(pos_img))
        neg_img_tensor = self.img_transform(Image.open(neg_img))

        return (
            audio, 
            pos_img_tensor, 
            neg_img_tensor, 
            pos_label,
            self.img_class_list[pos_label]
        )
        
    def _load_item(self, path):
        waveform, sample_rate = torchaudio.load(path)
        return waveform[0], sample_rate

    def _resample_item(self, wav, org_sr):
        _resample = Resample(
            orig_freq=org_sr,
            new_freq=self.SAMPLING_RATE,
        )
        return _resample(wav)

    def _get_random_wav_seg(self, wav):
        if wav.shape[0] > self.WAV_INPUT_LENGTH:
            _start = random.randint(0, wav.shape[0] - self.WAV_INPUT_LENGTH)
            wav = wav[_start:_start + self.WAV_INPUT_LENGTH]
        else:
            wav = F.pad(wav, pad=(0, self.WAV_INPUT_LENGTH - wav.shape[0]), mode='constant', value=0)
        return wav

    def _get_sequential_wav_seg_list(self, wav):
        if wav.shape[0] > self.WAV_INPUT_LENGTH:
            num_segs = wav.shape[0] // self.WAV_INPUT_LENGTH
            wavs = [wav[i*self.WAV_INPUT_LENGTH:(i+1)*self.WAV_INPUT_LENGTH] for i in range(num_segs)]
        else:
            wav = F.pad(wav, pad=(0, self.WAV_INPUT_LENGTH - wav.shape[0]), mode='constant', value=0)
            wavs = [wav]
        return wavs

    def __len__(self) -> int:
        return len(self.audio_paths)

    def _get_pos_neg_inp(self, idx):
        pos_label = self.audio_labels[idx]
        pos_img = random.sample(self.label_idx_to_img_paths_dict[pos_label], 1)[0]
        neg_label = random.choice([x for x in list(self.audio_label_set) if x != pos_label])
        neg_img = random.sample(self.label_idx_to_img_paths_dict[neg_label], 1)[0]
        
        pos_img = self.img_transform(Image.open(pos_img))
        neg_img = self.img_transform(Image.open(neg_img))

        return pos_img, neg_img