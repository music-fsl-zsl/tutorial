# Zero-Shot Learning Foundations

Zero-shot learning is yet another approach for classifying the classes that are not observed during training. Main difference from few-shot learning is that it does not require any additional label data for novel class inputs. 
Therefore, in zero-shot learning, there is no further training step for unseen classes. Instead, during the training phase, the model learns how to use the side information that can potentially cover the relationship between any of both seen and unseen classes. After training, it can handle the cases where inputs from unseen classes are to be classified.
ZSL was originally inspired by human’s ability to recognize objects without seeing training examples or even create new categories dynamically based on semantic analysis. The semantic analysis comes from the prior knowledge where general relationship between seen and unseen classes are learned.



# Overview on Zero-shot Learning Paradigm

<img src = "../assets/zsl/zsl_01.png" width=700>

This is enabled by utilizing an auxiliary information that provides a separate semantic space of the label classes.

<img src = "../assets/zsl/zsl_02.png" width=800>

Let's look into a case of an audio-based instrument classication task. First, given training audio and their associated class labels (seen classes), we train a classifier that projects input vectors onto the audio embedding space. 

<img src = "../assets/zsl/zsl_process_01.svg" width=900>

However, there isn't a way to make prediction of unseen labels for unseen audio inputs yet.

<img src = "../assets/zsl/zsl_process_02.svg" width=900>

As forementioned we use the side information that can inform the relationships between both seen and unseen labels. 
There are various sources of the side information, such as class-attribute vectors infered from an annotated dataset, or general word embedding vectors trained on a large corpus of documents. We will go over in detail in the later section.  

<img src = "../assets/zsl/zsl_process_03.svg" width=900>

The core of zero-shot learning paradigm is to learn the compatibility function between the embedding space of the inputs and the side information space of their labels. 
- Compatibility function : $F(x, y ; W)=\theta(x)^T W \phi(y)$
    - $\theta(x)$ : input embedding function.
    - $W$ : mapping function. 
    - $\phi(y)$ : label embedding function.

<img src = "../assets/zsl/zsl_process_04.svg" width=900>

A typical approach is to train a mapping function between the two.
By unveiling relationship between the side information space and the our input feature space, it is possible to map vectors from one space to the other.

<img src = "../assets/zsl/zsl_process_05.svg" width=900>


After training, arbitrary inputs of unseen labels can be predicted to the corresponding class. 

<img src = "../assets/zsl/zsl_process_06.svg" width=900>

Another option is to train a separate zero-shot embedding space where the embeddings from both spaces are projected (a metric-learning approach).

- E.g. Training mapping functions $W_1$ and $W_2$ with a pairwise loss function : $\sum_{y \in \mathcal{Y}^{seen}}\left[\Delta\left(y_n, y\right)+F\left(x_n, y ; W_1, W_2\right)-F\left(x_n, y_n ; W_1, W_2\right)\right]_{+}$
    - where $F(x, y ; W_1, W_2)= -\text{Distance}(\theta(x)^T W_1, \phi(y)^T W_2)$

<img src = "../assets/zsl/zsl_process_07.svg" width=900>

In this case, the inputs and the classes are projected onto another zero-shot embedding space.

<img src = "../assets/zsl/zsl_process_08.svg" width=900>


<img src = "../assets/zsl/emb_space_zsl.png" width=500>

Different strategies for ZSL embedding space [ ].


So far, we've gone throught the broad concept of zero-shot learning paradigm. 
Next, we'll look more into the detailed task formulation.



