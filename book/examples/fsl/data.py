from typing import Optional, Callable, Dict
import random

from torch.utils.data import Dataset
from util import load_excerpt

class EpisodicDataset:
    """
    A wrapper for single-class datasets that loads few-shot learning episodes. 

    """

    def __init__(self,
        dataset_map: Dict[str, Dataset], 
        n_way: int = 2, 
        n_support: int = 5,
        n_query: int = 7,
        n_episodes: int = 100,
        duration: float = 1.0, 
        sample_rate: int = 16000,
        transform: Optional[Callable] = None,
    ):
        self.classlist = list(dataset_map.keys())
        self.dataset_map = dataset_map

        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        self.n_episodes = n_episodes
        self.duration = duration
        self.sample_rate = sample_rate
        self.transform = transform
    
    def __getitem__(self, index):
        # make episodes returned from this index deterministic
        rng = random.Random(index)

        # sample the list of classes for this episode
        classes = rng.sample(self.classlist, self.n_way)

        # sample the support and query sets for this episode
        support, query = [], []
        for c in classes:
            # grab the dataset for this class
            dataset = self.dataset_map[c]

            # sample the support and query sets for this class
            indices = rng.sample(range(len(dataset)), self.n_support + self.n_query)
            items = [dataset[i] for i in indices]

            # add the class label to each item
            for item in items:
                item["label"] = self.classlist.index(c)

            # split the support and query sets
            support.extend(items[:self.n_support])
            query.extend(items[self.n_support:])

        # apply the transform to the support and query sets
        if self.transform is not None:
            support = self.transform(support)
            query = self.transform(query)
        
        return support, query

    def __len__(self):
        return self.n_episodes

import mirdata

class TinySOL:
    """a class-conditional wrapper for the TinySOL dataset

    Instances of this dataset will only load tracks for a single instrument.
    """

    dataset = mirdata.initialize('tinysol')
    dataset.download()

        # get the instrument classes for tinysol
    instruments = list(
        set([track.instrument_full 
            for track in dataset.load_tracks().values()
        ])
    )

    def __init__(self, 
            instrument: str,
            duration: float = 0.5, 
            sample_rate: int = 16000,
        ):
        self.instrument = instrument
        self.duration = duration
        self.sample_rate = sample_rate

        # load all tracks for this instrument
        self.tracks = []
        for track in self.dataset.load_tracks().values():
            if track.instrument_full == instrument:
                self.tracks.append(track)

    def __getitem__(self, index):
        # load the track for this index
        track = self.tracks[index]

        # load the excerpt
        excerpt = load_excerpt(track.audio_path, self.duration, self.sample_rate)

        return excerpt

    def __len__(self):
        return len(self.tracks)


# def episode_collate(episodes: List):
#     support = {}
#     query = {}

#     # find the keys in a subset
#     subset_keys = episodes[0][0][0].keys()

#     # collect the support and query sets
    
                




if __name__ == "__main__":
    from torch.utils.data import DataLoader

    # create a dataset for each instrument
    dataset_map = {
        instrument: TinySOL(instrument) 
            for instrument in TinySOL.instruments
    }

    # create an episodic dataset
    episodic_dataset = EpisodicDataset(dataset_map)

    # create a dataloader
    dataloader = DataLoader(episodic_dataset, batch_size=16, shuffle=True)

    batch = next(iter(dataloader))
    breakpoint()






