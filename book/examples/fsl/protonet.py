
import torch
from torch import nn

class EpisodicBatchWrap(nn.Module):

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, batch):
        n_batch, n_episodes = batch.shape[0:2]
        batch = batch.view(n_batch * n_episodes, *batch.shape[2:])
        output = self.module(batch)
        output = output.view(n_batch, n_episodes, *output.shape[1:])
        return output


class PrototypicalNet(nn.Module):

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
    
    def forward(self, support, query):
        # compute the embeddings for the support and query sets
        support["embeddings"] = self.backbone(support["audio"])
        query["embeddings"] = self.backbone(query["audio"])

        # group the support embeddings by instrument
        support_embeddings = {}
        for c in set(support["label"]):
            indices = (support["label"] == c).nonzero().squeeze(-1)
            support_embeddings[c] = support["embeddings"][indices]

        # compute the prototypes for each class
        prototypes = torch.stack([
            support_embeddings[k].mean(dim=0)
            for k in support_embeddings.keys()
        ])

        # compute the distances between each query and prototype
        distances = torch.cdist(
            query["embeddings"].unsqueeze(0), 
            prototypes.unsqueeze(0),
            p=2
        ).squeeze(0)

        # square the distances to get the sq euclidean distance
        distances = distances ** 2
        logits = -distances

        # return the logits, so we can use torch.logsoftmax 
        # for higher numerical stability during training
        return logits


if __name__ == "__main__":
    from backbone import Backbone
    backbone = Backbone(sample_rate=16000)
    protonet = PrototypicalNet(backbone)
    print(protonet)

    support = {
        "audio": torch.randn(10, 1, 16000),
        "target": torch.randint(0, 10, (10,))
    }
    query = {
        "audio": torch.randn(10, 1, 16000),
        "target": torch.randint(0, 10, (10,))
    }
    
    print(protonet(support, query))
    breakpoint()
    
    