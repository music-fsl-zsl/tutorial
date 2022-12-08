# Introduction: Zero-Shot Learning in PyTorch

In this coding tutorial, we will train a zero-shot instrument classifier that can predict unseen classes by using two different types of side information. 

The content will focus on experiencing the zero-shot classification procedure. Again, we're using [TinySOL dataset](https://zenodo.org/record/3685367) for the instrument audio data and labels for its small size and ease of accessibility. 

Both experiments will be conducted based on a similar siamese network architecture which trains a common embedding space where the audio and the side information embeddings are projected to be further compared. 

## (1) GloVe word embeddings as side information 

In the first experiment, we'll use [GloVe word embeddings](https://nlp.stanford.edu/projects/glove/) as the side information.

```{figure} ../assets/zsl/zsl_coding_ex02.png
:width: 900px
```

## (2) Image feature embeddings as side information 

In the second experiment, we'll use a pretrained image classfication model along with [PPMI dataset](https://ai.stanford.edu/~bangpeng/ppmi.html) as the side information. Being trained with a large image data corpus, the image classification model extracts general visual embeddings from the instrument images from PPMI dataset. 

```{figure} ../assets/zsl/zsl_coding_ex01.png
:width: 900px
```

A repo with ready-to-go code for this tutorial can be found on [github](https://github.com/music-fsl-zsl/music_zsl). Feel free to use it as a starting point for your own zero-shot experiment.

### Requirements

```
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

Here are the links to each section. 

1. [**Data preparation**](/zsl-example/data_prep.html): we'll learn how to prepare splits for zero-shot experiments.
2. [**Models and data I/O**](/zsl-example/model.html): we'll learn how to code the simese architecture, the training procedure, and the prediction procedure.  
3. [**Training the Word-Audio ZSL model**](/zsl-example/zsl_training_word_audio.html): we'll check on the training code for the word-audio zero-shot model.
4. [**Evaluating the Word-Audio ZSL model**](/zsl-example/zsl_eval_word_audio.html): we'll learn how to run the zero-shot evaluation and investigate the results of the word-audio zero-shot model.
5. [**Training Image-Audio ZSL model**](/zsl-example/zsl_training_image_audio.html): we'll check on the training code for the image-audio zero-shot model.
6. [**Evaluating the Image-Audio ZSL model**](/zsl-example/zsl_eval_image_audio.html): we'll learn how to run the zero-shot evaluation and investigate the results of the image-audio zero-shot model.


