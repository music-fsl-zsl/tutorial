# Training a Few-Shot Instrument Classifier

In this coding tutorial, we will train a musical instrument classifier that is able to categorize novel, unseen classes, using few-shot learning. 

This tutorial  familiar with [PyTorch](https://pytorch.org/), as well as the fundamentals of supervised machine learning and music signal processing. 
The content will focus on introducing the few-shot learning paradigm, and how to use it to train an audio classifier for unseen musical instrument classes. 

```{figure} ../assets/sample-episode.png
---
name: sample-episode
---
By the end of this tutorial, you will be able to train a prototypical network for few-shot learning on unseen musical instrument sounds. During the last part of the tutorial, you will be able to plot the embedding space of an episode using T-SNE, and visualize the learned prototypes for a sample test episode.
```

We'll train a [Prototypical Network](/foundations-fsl/metric-based-fsl/) on the [TinySOL dataset](https://zenodo.org/record/3685367) on a 5-way, 5-shot classification task. 
Note that, in this tutorial, we are using the TinySOL dataset because of its relatively small size and ease of accessibility, thanks to the [mirdata](https://github.com/mir-dataset-loaders/mirdata/) library. 

However, the same principles can be applied to any dataset. For a bigger challenge, try using the [MedleyDB](https://medleydb.weebly.com/) dataset, which contains a larger number of instrument classes. 

A repo with ready-to-go code from this tutorial can be found on [github](https://github.com/music-fsl-zsl/tutorial/fsl-example).

### Requirements

See [requirements.txt](https://github.com/music-fsl-zsl/tutorial/book/fsl-example/fsl/requirements.txt)

### Table of Contents

Here's a rundown of the topics we'll cover:

1. [**Datasets**](/fsl-example/datasets.md): we'll learn how to create an Episodic Sampler for the TinySOL dataset, and use it to create a few-shot learning dataset.
2. [**Models**](/fsl-example/models.md): we'll learn how to create a Prototypical Network, given any backbone model architecture.
3. [**Training**](/fsl-example/training.md): we'll learn how to train our few-shot model using Pytorch Lightning.
4. [**Evaluation**](/fsl-example/evaluation.md): we'll learn how to evaluate our few-shot model on the test set, and the embedding space for some of the test episodes.
