from pathlib import Path

import numpy as np
import torch
import tqdm

from  music_fsl.protonet import PrototypicalNet
from  music_fsl.backbone import Backbone
from  music_fsl.train import build_datasets, FewShotLearner
from  music_fsl.util import dim_reduce, embedding_plot, batch_device

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate(
        output_dir: Path,
        checkpoint_path: str,
    ):
    output_dir.mkdir(exist_ok=True, parents=True)

    _, test_data = build_datasets()

    learner = FewShotLearner(
        protonet=PrototypicalNet(
            backbone=Backbone(16000),
        ),
    )
    learner.to(DEVICE)

    checkpoint = torch.load(checkpoint_path)
    learner.load_state_dict(checkpoint['state_dict'])

    learner.eval()

    embedding_table = []

    pbar = tqdm.tqdm(range(len(test_data)))
    for episode_idx in pbar:
        support, query = test_data[episode_idx]
        batch_device(support, DEVICE)
        batch_device(query, DEVICE)

        # get the embeddings
        logits = learner.protonet(support, query)

        for subset_idx, subset in enumerate((support, query)):
            for emb, label in zip(subset["embeddings"], subset["label"]):
                embedding_table.append({
                    "embedding": emb.detach().cpu().numpy(),
                    "label": support["classlist"][label],
                    "marker": ("support", "query")[subset_idx], 
                    "episode_idx": episode_idx
                })
            
        for class_idx, emb in enumerate(support["prototypes"]):
            embedding_table.append({
                "embedding": emb.detach().cpu().numpy(),
                "label": support["classlist"][class_idx],
                "marker": "prototype", 
                "episode_idx": episode_idx
            })
        

    embeddings = dim_reduce(
        embeddings=np.stack([d["embedding"] for d in embedding_table]),
        method="tsne",
        n_components=2,
    )
    for entry, dim_reduced_embedding in zip(embedding_table, embeddings):
        entry["embedding"] = dim_reduced_embedding

    embeddings_dir = (output_dir / "embeddings")
    embeddings_dir.mkdir(exist_ok=True)
    for episode_idx in range(len(test_data)):
        subtable = [d for d in embedding_table if d["episode_idx"] == episode_idx]

        fig = embedding_plot(
            proj=np.stack([d["embedding"] for d in subtable]),
            color_labels=[d["label"] for d in subtable],
            marker_labels=[d["marker"] for d in subtable],
            title=f"embeddings",
        )
        
        fig.write_image(embeddings_dir / f"episode-{episode_idx}.png")


if __name__ == "__main__":
    
    evaluate(
        output_dir=Path("logs/version_10/eval"),
        checkpoint_path='logs/version_10/checkpoints/epoch=0-step=1400.ckpt', 
    )
