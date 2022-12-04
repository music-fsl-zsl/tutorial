# Introduction: Few-Shot Learning in PyTorch

In this coding tutorial, we will train a musical instrument classifier that is able to categorize novel, unseen classes using few-shot learning. 

This tutorial assumes the reader is familiar with [PyTorch](https://pytorch.org/), as well as the fundamentals of supervised machine learning and music signal processing. 
The content will focus on introducing the few-shot learning paradigm and how to use it to train an audio classifier for unseen musical instrument classes.

```{figure} ../assets/sample-episode.png
---
name: sample-episode
---
By the end of this tutorial, you will be able to train a prototypical network for few-shot learning on unseen musical instrument sounds. During the last part of the tutorial, you will be able to plot the embedding space of an episode using T-SNE, and visualize the learned prototypes for a sample test episode.
```

We'll train a [Prototypical Network](/foundations-fsl/metric-based-fsl/) on the [TinySOL dataset](https://zenodo.org/record/3685367) on a 5-way, 5-shot classification task. 
Note that in this tutorial, we are using the TinySOL dataset because of its relatively small size and ease of accessibility, thanks to the [mirdata](https://github.com/mir-dataset-loaders/mirdata/) library. 

However, the same principles can be applied to any dataset. For a bigger challenge, you can try using the [MedleyDB](https://medleydb.weebly.com/) dataset, which contains a larger number of instrument classes. 

## Running the Code 

### On your machine

A repo with ready-to-go code for this tutorial can be found on [github](https://github.com/music-fsl-zsl/music_fsl). Feel free to use it as a starting point for your own few-shot MIR projects! :) 

### On Google Colab

You can also run the code on Google Colab. 

If any chapter in this book has an interactive component, you will see a rocket icon on the top right corner of the page. Clicking on the icon will give you the option to run the code on Google Colab.

[Google Colab](https://colab.research.google.com) is a cloud-based Jupyter Notebook offered by Google. To access it, click on the rocket icon and select "Colab" from the drop-down menu. This will open a populated Colab notebook in your browser. Colab offers free access to GPUs, which can be used to speed up certain tasks. To use the free GPUs, go to "Edit > Notebook Settings" and select GPU from the Hardware Accelerator drop-down menu. 

### Requirements

```
mirdata
librosa

torch
numpy
torchaudio
torchmetrics
pytorch-lightning
tensorboard

torchvision
sklearn
umap-learn
pandas
plotly
kaleido
```

### Table of Contents

Here's a rundown of the topics we'll cover:

1. [**Datasets**](/fsl-example/datasets): We'll learn how to create a class-conditional dataset for few-shot learning, using the TinySOL dataset.
2. [**Episodes**](/fsl-example/episodes): We'll learn how to construct few-shot learning episodes from a dataset, using an episode dataset.  
3. [**Models**](/fsl-example/models): We'll learn how to create a Prototypical Network given any backbone model architecture.
4. [**Training**](/fsl-example/training): We'll learn how to train our few-shot model using Pytorch Lightning.
5. [**Evaluating//Visualizing**](/fsl-example/evaluating): We'll learn how to evaluate our few-shot model on the test set, and the embedding space for some of the test episodes.
[]()

