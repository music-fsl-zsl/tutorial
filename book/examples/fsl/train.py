
import torch
import numpy as np
from torch import nn
import pytorch_lightning as pl
from torchmetrics import Accuracy

from backbone import Backbone
from data import TinySOL, EpisodicDataset
from protonet import PrototypicalNet
from util import split, dim_reduce, plotly_fig_to_tensor

class FewShotLearner(pl.LightningModule):

    def __init__(self, 
        protonet: nn.Module, 
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.protonet = protonet
        self.learning_rate = learning_rate

        self.loss = nn.CrossEntropyLoss()
        self.metrics = nn.ModuleDict({
            'accuracy': Accuracy()
        })

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def step(self, batch, batch_idx, tag: str):
        support, query = batch

        logits, _, _ = self.protonet(support, query)
        loss = self.loss(logits, query["label"])

        output = {"loss": loss}
        for k, metric in self.metrics.items():
            output[k] = metric(logits, query["label"])

        for k, v in output.items():
            self.log(f"{k}/{tag}", v)
        return output

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "train")
    
    def validation_step(self, batch, batch_idx):
        output = self.step(batch, batch_idx, "val")
        if batch_idx ==  0:
            self.save_sample(batch)
        return output

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "test")

    def save_sample(self, batch):
        support, query = batch
        logits = self.protonet(support, query)

        all_embeddings = []
        for subset_idx, subset in enumerate((support, query)):
            for emb, label in zip(subset["embeddings"], subset["label"]):
                all_embeddings.append({
                    "embedding": emb.detach().cpu().numpy(),
                    "label": support["classlist"][label],
                    "marker": ("support", "query")[subset_idx]
                })
            
        for emb, label in zip(query["embeddings"], query["label"]):
            all_embeddings.append({
                "embedding": emb.detach().cpu().numpy(),
                "label": support["classlist"][label],
                "marker": "prototype"
            })
        
        fig = dim_reduce(
            embeddings=np.stack([d["embedding"] for d in all_embeddings]),
            color_labels=[d["label"] for d in all_embeddings],
            marker_labels=[d["marker"] for d in all_embeddings],
            n_components=2,
            title=f"embeddings/step-{self.global_step}",
            method="umap"
        )

        image = plotly_fig_to_tensor(fig)
        self.logger.experiment.add_image("embeddings", image, self.global_step)


TRAIN_INSTRUMENTS = [
    'French Horn', 
    'Violin', 
    'Flute', 
    'Contrabass', 
    'Trombone', 
    'Cello', 
    'Clarinet in Bb', 
    'Oboe',
    'Accordion'
]

TEST_INSTRUMENTS = [
    'Bassoon',
    'Viola',
    'Trumpet in C',
    'Bass Tuba',
    'Alto Saxophone'
]

def build_datasets(
        n_train_episodes: int = int(100e3), 
        n_val_episodes: int = 100, 
        n_way=5, 
        sample_rate: int = 16000,
    ):

    # create a dataset for each instrument
    train_datasets = {
        instrument: TinySOL(instrument, sample_rate=sample_rate)
            for instrument in TRAIN_INSTRUMENTS
    }

    val_datasets = {
        instrument: TinySOL(instrument, sample_rate=sample_rate)
            for instrument in TEST_INSTRUMENTS
    }

    # create an episodic dataset
    train_data = EpisodicDataset(
        train_datasets,
        n_way=n_way,
        n_episodes=n_train_episodes, 
    )
    val_data = EpisodicDataset(
        val_datasets, 
        n_way=n_way,
        n_episodes=n_val_episodes,
    )

    return train_data, val_data

def train(
        sample_rate: int = 16000,
        num_workers: int = 10,
    ):
    # create the datasets
    train_data, val_data = build_datasets(sample_rate=sample_rate)

    # dataloaders
    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        train_data, batch_size=None, 
        shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_data, batch_size=None, 
        shuffle=True, num_workers=num_workers
    )
    
    # build models
    backbone = Backbone(sample_rate=sample_rate)
    protonet = PrototypicalNet(backbone)
    learner = FewShotLearner(protonet)
    print(learner)

    # set up the trainer
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.profiler import SimpleProfiler

    trainer = pl.Trainer(
        gpus=1 if torch.cuda.is_available() else 0,
        max_epochs=1,
        log_every_n_steps=1, 
        val_check_interval=50,
        profiler=SimpleProfiler(
            filename="profile.txt",
        ), 
        logger=TensorBoardLogger(
            save_dir=".",
            name="logs"
        ), 
    )

    # train!
    trainer.fit(learner, train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    train()
