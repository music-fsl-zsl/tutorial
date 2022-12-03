from typing import Optional, Callable, Dict, List, Any, Tuple
import random
from collections import defaultdict

import torch
import numpy as np
import mirdata
from torch.utils.data import Dataset
import music_fsl.util as util


class ClassConditionalDataset(torch.utils.data.Dataset):

    def __getitem__(self, index: int) -> Dict[Any, Any]:
        """
        Grab an item from the dataset. The item returned must be a dictionary. 
        """
        raise NotImplementedError
    
    @property
    def classlist(self) -> List[str]:
        """
        The classlist property returns a list of class labels available in the dataset.
        This property enables users of the dataset to easily access a list of all the classes in the dataset.

        Returns:
            List[str]: A list of class labels available in the dataset. 
        """
        raise NotImplementedError

    @property
    def class_to_indices(self) -> Dict[str, List[int]]:
        """
        Returns a dictionary where the keys are class labels and the values are 
        lists of indices in the dataset that belong to that class. 
        This property enables users of the dataset to easily access 
        examples that belong to specific classes. 

        Implement me!

        Returns:
            Dict[str, List[int]]: A dictionary mapping class labels to lists of dataset indices. 
        """
        raise NotImplementedError


class TinySOL(ClassConditionalDataset):
    """
    Initialize a `TinySOL` dataset instance.

    Args:
        instruments (List[str]): A list of instruments to include in the dataset.
        duration (float): The duration of each audio clip in the dataset (in seconds).
        sample_rate (int): The sample rate of the audio clips in the dataset (in Hz).
    """

    INSTRUMENTS = [
        'Bassoon', 'Viola', 'Trumpet in C', 'Bass Tuba',
        'Alto Saxophone', 'French Horn', 'Violin', 
        'Flute', 'Contrabass', 'Trombone', 'Cello', 
        'Clarinet in Bb', 'Oboe', 'Accordion'
    ]

    def __init__(self, 
            instruments: List[str] = None,
            duration: float = 1.0, 
            sample_rate: int = 16000,
        ):
        if instruments is None:
            instruments = self.INSTRUMENTS

        self.instruments = instruments
        self.duration = duration
        self.sample_rate = sample_rate

        # initialize the tinysol dataset and download if necessary
        self.dataset = mirdata.initialize('tinysol')
        self.dataset.download()

        # make sure the instruments passed in are valid
        for instrument in instruments:
            assert instrument in self.INSTRUMENTS, f"{instrument} is not a valid instrument"

        # load all tracks for this instrument
        self.tracks = []
        for track in self.dataset.load_tracks().values():
            if track.instrument_full in self.instruments:
                self.tracks.append(track)


    @property
    def classlist(self) -> List[str]:
        return self.instruments

    @property
    def class_to_indices(self) -> Dict[str, List[int]]:
        # cache it in self._class_to_indices 
        # so we don't have to recompute it every time
        if not hasattr(self, "_class_to_indices"):
            self._class_to_indices = defaultdict(list)
            for i, track in enumerate(self.tracks):
                self._class_to_indices[track.instrument_full].append(i)

        return self._class_to_indices

    def __getitem__(self, index) -> Dict:
        # load the track for this index
        track = self.tracks[index]

        # load the excerpt
        data = util.load_excerpt(track.audio_path, self.duration, self.sample_rate)
        data["label"] = track.instrument_full

        return data

    def __len__(self) -> int:
        return len(self.tracks)



class EpisodeDataset(torch.utils.data.Dataset):
    """
        A dataset for sampling few-shot learning tasks from a class-conditional dataset.

    Args:
        dataset (ClassConditionalDataset): The dataset to sample episodes from.
        n_way (int): The number of classes to sample per episode.
            Default: 5.
        n_support (int): The number of samples per class to use as support.
            Default: 5.
        n_query (int): The number of samples per class to use as query.
            Default: 20.
        n_episodes (int): The number of episodes to generate.
            Default: 100.
    """
    def __init__(self,
        dataset: ClassConditionalDataset, 
        n_way: int = 5, 
        n_support: int = 5,
        n_query: int = 20,
        n_episodes: int = 100,
    ):
        self.dataset = dataset

        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        self.n_episodes = n_episodes
    
    def __getitem__(self, index: int) -> Tuple[Dict, Dict]:
        """Sample an episode from the class-conditional dataset. 

        Each episode is a tuple of two dictionaries: a support set and a query set.
        The support set contains a set of samples from each of the classes in the
        episode, and the query set contains another set of samples from each of the
        classes. The class labels are added to each item in the support and query
        sets, and the list of classes is also included in each dictionary.

        Yields:
            Tuple[Dict[str, Any], Dict[str, Any]]: A tuple containing the support
            set and the query set for an episode.
        """
        # seed the random number generator so we can reproduce this episode
        rng = random.Random(index)

        # sample the list of classes for this episode
        episode_classlist = rng.sample(self.dataset.classlist, self.n_way)

        # sample the support and query sets for this episode
        support, query = [], []
        for c in episode_classlist:
            # grab the dataset indices for this class
            all_indices = self.dataset.class_to_indices[c]

            # sample the support and query sets for this class
            indices = rng.sample(all_indices, self.n_support + self.n_query)
            items = [self.dataset[i] for i in indices]

            # add the class label to each item
            for item in items:
                item["target"] = torch.tensor(episode_classlist.index(c))

            # split the support and query sets
            support.extend(items[:self.n_support])
            query.extend(items[self.n_support:])

        # collate the support and query sets
        support = util.collate_list_of_dicts(support)
        query = util.collate_list_of_dicts(query)

        support["classlist"] = episode_classlist
        query["classlist"] = episode_classlist
        
        return support, query

    def __len__(self):
        return self.n_episodes

    def print_episode(self, support, query):
        """Print a summary of the support and query sets for an episode.

        Args:
            support (Dict[str, Any]): The support set for an episode.
            query (Dict[str, Any]): The query set for an episode.
        """
        print("Support Set:")
        print(f"  Classlist: {support['classlist']}")
        print(f"  Audio Shape: {support['audio'].shape}")
        print(f"  Target Shape: {support['target'].shape}")
        print()
        print("Query Set:")
        print(f"  Classlist: {query['classlist']}")
        print(f"  Audio Shape: {query['audio'].shape}")
        print(f"  Target Shape: {query['target'].shape}")



if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = TinySOL()

    # create an episodic dataset
    episodes = EpisodeDataset(dataset)

    # create a dataloader
    dataloader = DataLoader(episodes, batch_size=None, shuffle=True)

    batch = next(iter(dataloader))
    breakpoint()

