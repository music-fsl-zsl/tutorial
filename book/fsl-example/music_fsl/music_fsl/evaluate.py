from pathlib import Path

import numpy as np
import torch
import tqdm
from torchmetrics import Accuracy
import argbind

from music_fsl.protonet import PrototypicalNet
from music_fsl.backbone import Backbone
from music_fsl.train import FewShotLearner, TEST_INSTRUMENTS
from music_fsl.util import dim_reduce, embedding_plot, batch_device

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@argbind.bind(without_prefix=True)
def evaluate(
        checkpoint_path: str,
        n_way: int = 5,
        n_support: int = 5,
        n_query: int = 15,
        n_episodes: int = 100,
    ):
    """
    The evaluate function evaluates a pre-trained few-shot learning model on the TinySOL dataset. 

    Args:
        checkpoint_path (str): The path to the saved PyTorch Lightning checkpoint of the trained model.
        n_way (int): The number of classes to sample per episode.
            Default: 5.
        n_support (int): The number of samples per class to use as support.
            Default: 5.
        n_query (int): The number of samples per class to use as query.
            Default: 15.
        n_episodes (int): The number of episodes to evaluate on.
            Default: 100.

    The function loads the pre-trained model from the specified checkpoint and 
    evaluates it on the TinySOL test set of instrument classes. 
    It also saves a TSNE visualization of the evaluation results to the specified output directory.
    """
    output_dir = Path(checkpoint_path).parent.parent

    # load our evaluation data
    test_episodes = EpisodeDataset(
        dataset=TinySOL(
            instruments=TEST_INSTRUMENTS, 
            sample_rate=sample_rate
        ), 
        n_way=n_way, 
        n_support=n_support,
        n_query=n_query, 
        n_episodes=n_test_episodes
    )

    # load the few-shot learner from checkpoint
    learner = FewShotLearner.load_from_checkpoint(checkpoint_path)
    learner.eval()

    # instantiate the accuracy metric
    metric = Accuracy(num_classes=n_way, average="samples")

    # collect all the embeddings in the test set
    # so we can plot them later
    embedding_table = []
    pbar = tqdm.tqdm(range(len(test_episodes)))
    for episode_idx in pbar:
        support, query = test_episodes[episode_idx]

        # move all tensors to cuda if necessary
        batch_device(support, DEVICE)
        batch_device(query, DEVICE)

        # get the embeddings
        logits = learner.protonet(support, query)

        # compute the accuracy
        acc = metric(logits, query["label"])
        pbar.set_description(f"Accuracy: {acc.item():.2f}")

        # add all the support and query embeddings to our records
        for subset_idx, subset in enumerate((support, query)):
            for emb, label in zip(subset["embeddings"], subset["label"]):
                embedding_table.append({
                    "embedding": emb.detach().cpu().numpy(),
                    "label": support["classlist"][label],
                    "marker": ("support", "query")[subset_idx], 
                    "episode_idx": episode_idx
                })
            
        # also add the prototype embeddings to our records
        for class_idx, emb in enumerate(support["prototypes"]):
            embedding_table.append({
                "embedding": emb.detach().cpu().numpy(),
                "label": support["classlist"][class_idx],
                "marker": "prototype", 
                "episode_idx": episode_idx
            })

    # compute the total accuracy across all episodes
    total_acc = metric.compute()
    print(f"Total accuracy, averaged across all episodes: {total_acc:.2f}")
        
    # perform a TSNE over all embeddings in the test dataset
    embeddings = dim_reduce(
        embeddings=np.stack([d["embedding"] for d in embedding_table]),
        method="tsne",
        n_components=2,
    )
    for entry, dim_reduced_embedding in zip(embedding_table, embeddings):
        entry["embedding"] = dim_reduced_embedding

    # plot the embeddings for each episode
    embeddings_dir = (output_dir / "embeddings")
    embeddings_dir.mkdir(exist_ok=True)
    for episode_idx in range(len(test_data)):
        subtable = [d for d in embedding_table if d["episode_idx"] == episode_idx]

        fig = embedding_plot(
            proj=np.stack([d["embedding"] for d in subtable]),
            color_labels=[d["label"] for d in subtable],
            marker_labels=[d["marker"] for d in subtable],
            title=f"episode {episode_idx} -- embeddings",
        )
        
        fig.write_image(embeddings_dir / f"episode-{episode_idx}.png")

if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        evaluate()
