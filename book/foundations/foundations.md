# Few-Shot Learning Foundations

It should come with no surprise that, like any other machine learning problem, data is at the core of few-shot learning problems. This chapter looks at the foundations of few-shot learning -- namely how we think about (and structure) our data, when trying to learn new things with very little data for novel, unseen classes.

When solving traditional classification problems, we typically consider a closed set of classes. That is, we expect to see the same set of classes during inference as we did during training. Few-shot learning breaks that assumption, and instead expects that the classification model will encounter novel classes during inference, with one caveat: that there are a **few labeled examples** for each novel class at inference time. 

## Let's define the problem!

Consider that we would like to classify between $C$ classes, and we have exactly $K$ labeled examples that for each of those classes. 
We say that few-shot models are trained to solve a $C$-way, $K$-Shot classification task. 


```{figure} ../assets/foundations/support-query.png
---
name: support-query
---
A few-shot learning problem splits data into two separate sets: the support set (the few labeled examples of novel data) and the query set (the data we want to label).
```


Few shot learning tasks divide the few labeled data we have and the many unlabeled data we would like to to label into two separate subsets: the **support set** and the **query set**. 

The small set of labeled examples we are given at inference time is the **support set**. The support set is small in size and contains a few ($K$) examples for each of the classes we would like to consider. 

Formally, we define the support set as a set of labeled training pairs $S = \{(x_1, y_1,), (x_2, y_2), ..., (x_N, y_N)\}$, where:

- $x_i \in \mathbb{R}^D$ is a $D$-dimensional input vector.
- $y_i \in \{1,...,C\}$ is the class label that corresponds to $x_i$.
- $S_c$ refers to the set of examples with label $c$.
- $N$ is the size of the support set, where $N = C \times K$.  

On the other hand, the query set contains all of the examples we would like to label, typically denoted as $Q$.

### The goal

The goal of few-shot learning algorithms is to learn a classification model $f_\theta$ that is able to generalize to $C$ previously unseen classes at inference time, with just a few examples  $K$ for each previously unseen class.

## Meta Learning -- Learning to Learn

In order for a classifier to be able to learn a novel class with only a few labeled examples, we can employ a technique known as **meta learning**, or learning to learn.

```{note}
Even though our goal in few shot learning is to be able to learn novel classes with very few labeled examples, we *still* require a sizable training dataset with thousands of examples. The idea is that we can *learn how to learn new classes* from this large training set, and then apply that knowledge to learn novel classes with very few labeled examples.
```

### Episodic Training

In meta learning, we take our training dataset (as well as our evaluation dataset), and sample *episodes* from it. An episode is like a simulation of a few-shot learning scenario, typically with $C$ classes and $K$ labeled examples for each class -- similar to what we expect the model to be able to infer at inference time. Training a deep model by sampling few-shot learning episodes from a large training dataset is known as **episodic training**.  
% TODO: add citations here


```{note}
During episodic training, our model will see a completely new $K$-shot, $C$-way classification task at each step. To build a single episode, we must sample a completely new support set and query set during each training step.
Practically, this means that, for each episode, we have to choose a subset of $C$ classes from our training dataset, and then sample $K$ labeled examples (for the support set) and $q$ examples (for the query set) for each class that we randomly sampled. 
```

#### Evaluating a Few-Shot Model
Validation and Evaluation during episodic training can be done in a similar fashion to training. We can build a series of episodes from our validation and evaluation datasets, and then evaluate the model on each episode using standard classifcation metrics, like [idunnolistsomemetricsreferencehere]. 
% TODO (add reference to metrics above)

```{note}
Because our goal in few-shot learning is to be able to learn *new, unseen* classes, we want to make sure that the classes present in the training set and do **not** overlap with the classes used for validation and evaluation. This means that we must create train/validation/test splits differently than we would for a traditional supervised learning problem, where we would expect to see the same classes in the training, validation and evaluation sets.
```

% TODO: improve this figure
```{figure} ../assets/foundations/episodic-training.png
---
name: episodic-training
---
Episodic training is an efficient way of leveraging a large training dataset to train a few-shot learning model. **TODO**: improve this figure. 
```

We've now covered the basic foundations of few-shot learning. In the next chapter, we'll look at some of the most common approaches to few-shot learning, namely **metric**-based, **model**-based, and **optimization**-based approaches. 
