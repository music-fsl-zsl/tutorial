import torch
from torch import nn

class PrototypicalNet(nn.Module):

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
    
    def forward(self, support: dict, query: dict):
        """
        Forward pass through the protonet. 

        Args:
            support (dict): A dictionary containing the support set. 
                The support set dict must contain the following keys:
                    - audio: A tensor of shape (n_support, n_channels, n_samples)
                    - label: A tensor of shape (n_support) with label indices
                    - classlist: A tensor of shape (n_classes) containing the list of classes in this episode
            query (dict): A dictionary containing the query set.
                The query set dict must contain the following keys:
                    - audio: A tensor of shape (n_query, n_channels, n_samples)
        
        Returns:
            logits (torch.Tensor): A tensor of shape (n_query, n_classes) containing the logits

        After the forward pass, the support dict is updated with the following keys:
            - embeddings: A tensor of shape (n_support, n_features) containing the embeddings
            - prototypes: A tensor of shape (n_classes, n_features) containing the prototypes
        
        The query dict is updated with
            - embeddings: A tensor of shape (n_query, n_features) containing the embeddings

        """
        # compute the embeddings for the support and query sets
        support["embeddings"] = self.backbone(support["audio"])
        query["embeddings"] = self.backbone(query["audio"])

        # group the support embeddings by class
        support_embeddings = []
        for idx in range(len(support["classlist"])):
            embeddings = support["embeddings"][support["target"] == idx]
            support_embeddings.append(embeddings)
        support_embeddings = torch.stack(support_embeddings)

        # compute the prototypes for each class
        prototypes = support_embeddings.mean(dim=1)
        support["prototypes"] = prototypes

        # compute the distances between each query and prototype
        distances = torch.cdist(
            query["embeddings"].unsqueeze(0), 
            prototypes.unsqueeze(0),
            p=2
        ).squeeze(0)

        # square the distances to get the sq euclidean distance
        distances = distances ** 2
        logits = -distances

        # return the logits
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
    
    