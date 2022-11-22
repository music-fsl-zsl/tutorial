
# Zero-shot learning task formulation

Basic task formulation of zero-shot learning frames work is as follows.

Given a training dataset $\mathcal{S}={\left(x_n, y_n\right)}$, where the input is a $D$-dimensional feature vector. We first split the class labels into seen and unseen groups ($\mathcal{Y}^{seen}$, $\mathcal{Y}^{unseen}$). Only seen labels are used in training the zero-shot learning models. At test time, we evaluate the performance of the model for prediction of unseen label classes. 

## Basic formulation

To be more specific, 

$\mathcal{S}=\left\{\left(x_n, y_n\right), n=1 \ldots N\right\}$, with $x_n \in \mathcal{X}$ and $y_n \in \mathcal{Y}^{seen}$, where
- $\mathcal{S}$ refers to the set of pairs of input vectors and annotated classes.
- $\mathcal{Y}$ is the set of classes.
- $\mathcal{X}$ is the set of input vectors. 
- $x_n \in \mathbb{R}^D$ is a $D$-dimensional input vector.
- $y_n \in \{1,...,C\}$ is the class label that corresponds to $x_n$.
- $N$ is the size of the seen training pairs.  


During training phase, we learn $f: \mathcal{X} \rightarrow \mathcal{Y}$ by minimizing regularized loss function:

$$
\frac{1}{N} \sum_{n=1}^N L\left(y_n, f\left(x_n ; W\right)\right)+\Omega(W)
$$
, where 
- $L()=$ is a loss function
- $\Omega()=$ is a regularization term 
- $f(x ; W)=\arg \max _{y \in \mathcal{Y}} F(x, y ; W)$, where 
    - $F(x, y ; W)$ is a compatibility function that measures how compatible the input is with a class label.
    - $W$ is a learnable matrix (our model).
- or $f(x)=\underset{c}{\operatorname{argmax}} \prod_{m=1}^M \frac{p\left(a_m^c \mid x\right)}{p\left(a_m^c\right)}$. (Direct Attribute Projection (DAP))
    - $M$ : number of attributes
    - $a_m^c$ is the m-th attribute of class $c$ 
    - $p\left(a_m^c \mid x\right)$ is the attribute probability given input $x$ which is obtained from the attribute classifiers (our estimator).
    - $p\left(a_m^c\right)$ is the attribute prior estimated by the empirical mean of attributes over training classes. 

At the testing phase, a test input is labeled with an unseen class $\mathcal{Y}^{unseen} \subset \mathcal{Y}$ that results the maximum compatibility.

Training a zero-shot model can be acheived by

### maximizing the compatibility
- e.g. Linear compatibility function (learnable)
    - $F(x, y ; W)=\theta(x)^T W \phi(y)$
    - (can also be seen as learning a projection matrix that maximizes the dot product.)

or by
### minimizing a loss function
- Nonlinear mapping function (neural network layer $W_1$ and $W_2$) trained with a loss function
    <!-- - $F\left(x, y ; W_i\right)=\max _{1 \leq i \leq K} \theta(x)^T W_i \phi(y)$ -->
    - $\sum_{y \in \mathcal{Y}^{seen}} \sum_{x \in \mathcal{X}_y} \| \phi(y)-W_1 \tanh \left(W_2 \cdot \theta(x)\right) \|^2$ 
    - (distance metrics such as euclidean or cosine distance can be used.)


## Different training strategies
### (1) Learning by pairwise ranking of compatibility
DeViSE: A Deep Visual-Semantic Embedding Model (Frome et al., 2013)

Maximize the following objective function using pairwise ranking:

$$
\sum_{y \in \mathcal{Y}^{t r}}\left[\Delta\left(y_n, y\right)+F\left(x_n, y ; W\right)-F\left(x_n, y_n ; W\right)\right]_{+}
$$

- $\Delta\left(y_n, y\right)=1$ if $y_n=y$, otherwise 0
- Optimized by SGD


### (2) Learning by maximizing probability function 

CONSE (Norouzi et al., 2014)


Instead of learning the mapping function $f: \mathcal{X} \rightarrow \mathcal{Y}$ explicitly, learn a classifier from training inputs to seen labels. The probability of an input $\mathbf{x}$ belonging to a class label $y \in \mathcal{Y}_{seen}$ can then be estimated, denoted $p_{seen}(y \mid \mathbf{x})$, where $\sum_{y=1}^{n} p_{seen}(y \mid \mathbf{x})=1$.

- $f(x, t)$ : $\mathrm{t}^{th}$ most likely label for image $x$
    - $f(x, 1) \equiv \underset{y \in \mathcal{Y}_{seen}}{\operatorname{argmax}} p_0(y \mid \mathbf{x})$ : probability of an input $x$ belonging to a seen class:
- each class label $y(1 \leq y \leq n)$ is associated with a semantic embedding vector $s(y) \in \mathcal{S} \equiv \mathbb{R}^q$. 



Combination of semantic embeddings $(s)$ is used to assign an unknown image to an unseen class:

$$
\frac{1}{Z} \sum_{i=1}^T p_{t r}(f(x, t) \mid x) s(f(x, t))
$$

- $Z $: normalization factor given by $Z=\sum_{i=1}^T p_{t r}(f(x, t) \mid x)$
- $T$ : hyperparameter of controlling the maximum number of semantic embedding vectors to be considered.

If the classifier is confident in its prediction of a label $y$ for $\mathbf{x}$, i.e., $p_0(y \mid \mathbf{x}) \approx 1$, then $f(\mathbf{x}) \approx s(y)$. If not, predicted semantic embedding is somewhere between T most likely classes (weighted-sum).


### (3) Autoencoder approach

SAE (Kodirov et al., 2017)
Objective: similar to the linear auto-encoder

$$
\min _W\left\|\theta(x)-W^T \phi(y)\right\|^2+\lambda\|W \theta(x)-\phi(y)\|^2,
$$

- Learns a linear projection from $\theta(x)$ to $\phi(y)$
- Reconstruct the original input embedding by projection


<!-- 
## Embedding into common spaces
#### a common intermediate space
relationship of visual and semantic space  (by jointly exploring and exploiting a common intermediate space)
- CCA (canonical component analysis)
    - general kernal CCA method for learning semantic embeddings of web images and associated text
- 'Evaluation of output embeddings for fine-grained image classification'
    - image : attribute / text (word2vec) / hierarchical relationship (WordNet) → learn a joint embedding semantic space of the three modalities -->



<!-- ## LATEM (Xian et al., 2016)

Piecewise-linear compatibility

$F(x, y ; W)=\theta(x)^T W \phi(y)$
$F(x, y ; W)=\theta(x)^T W_i \phi(y)$ -->






<!-- 
### Embedding models 

- Bayesian models (early works)
    - prior knowledge of each attribute → make up for the limited supervision of novel class → generative model 
        - Direct Attribute Projection (DAP) / Indirect Attribute Projection (IAP) : ues SVM to learn embedding → Bayesian methods for classification (topic model / random forests)

- Semantic embedding 
    - learning to optimize precision at k of the ranked list of annotations for a given image 
    - learning a low-dimensional joint embedding space (for both images and annotations)
    - WSABIE (Web Scale Annotation by Image Embedding)



### Embedding into common spaces
#### a common intermediate space
relationship of visual and semantic space  (by jointly exploring and exploiting a common intermediate space)
- CCA (canonical component analysis)
    - general kernal CCA method for learning semantic embeddings of web images and associated text
- 'Evaluation of output embeddings for fine-grained image classification'
    - image : attribute / text (word2vec) / hierarchical relationship (WordNet) → learn a joint embedding semantic space of the three modalities
- 'Zero-Shot Object Recognition by Semantic Manifold Distance'
    - Fusing different types of semantic representations (in graph)
- semantic class label graph → fuse scores of different semantic representations
    - “Large-Scale Object Classification using Label Relation Graphs”
    - semantic label graph → fuse scores

### Deep embedding
- Use neural network to learn the relationship of visual and semantic space (by jointly exploring and exploiting a common intermediate space)
    - DeViSE
    - ConSE
- different loss functions
    - margin-based loss (DeViSE)
    - euclidean distance loss (“Predicting Deep Zero-Shot Convolutional Neural Networks using Textual Descriptions”)
        - visual space as embedding space
            - “Learning a Deep Embedding Model for Zero-Shot Learning”
        - semantic space as embedding space
        - Joint embedding space -->
