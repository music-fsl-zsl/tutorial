# Approaches

Now that we have a grasp of the foundations of few-shot learning, 
we'll take a look at some of the most common approaches to the solving few-shot problems. 

Recall that the goal of few-shot learning is to be able to learn to solve a new machine learning task given only a few labeled examples. In few-shot learning problems, we are given a small set of labeled examples for each class we would like to predict (the support set), as well as a larger set of unlabeled examples (the query set). We tend to refer to few-shot learning tasks as $K$-way, $N$-shot classification tasks, where $K$ is the number of classes we would like to predict, and $K$ is the number of labeled examples we are given for each class. 

When training a model to solve a few-shot learning task, we typically sample episodes from a large training dataset. An episode is a simulation of a few-shot learning task, where we sample $K$ classes and $N$ labeled examples for each class. Training a deep model by sampling few-shot learning episodes from a large training dataset is known as **episodic training**.

## Metric-based approaches

```{figure} ../assets/foundations/metric-based-learning.png
---
name: metric-based-learning
---

```

Metric-based approaches to few-shot learning are able to learn an embedding space where examples that belong to the same class are close together, even if the examples belong to classes that were not seen during training. 

Episodic training is essential to making metric-based few-shot models succeed in practice. Without episodic training, training a model using only $K$ examples for each class would result in poor generalization, and the model would not be able to generalize to new classes. 

At the center of metric-based few-shot learning approches is a similarity _metric_, which we will refer to as $g_{sim}$. 

Typically, we use this similarity metric to compare how similar examples in the query set are to examples in the support set. After knowing how similar a query example is to each example in the support set, we can infer to which class in the support set the query example belongs to. 

This similarity comparison is typically done in the embedding space of some neural net model, which we will refer to as $f_\theta$. Thus, during episodic training, we train $f_\theta$ to learn an embedding space where examples that belong to the same class are close together, and examples that belong to different classes are far apart. 

There are many different metric-based approaches to few-shot learning, and they all differ in how they define the similarity metric $g_sim$, and how they use it to compare query examples to support examples as well as formulate a training objective.

Among the most popular metric-based approaches are Prototypical Networks {cite}`snell2017prototypical`, Matching Networks {cite}`vinyals2016matching`, and Relation Networks {cite}`sung2018relation`.

### Example: Prototypical networks

```{figure} ../assets/foundations/prototypical-net.png
---
name: prototypical-net
---
The figure above illustrates a 5-shot, 3-way classification task between tambourine (red), maracas (green), and djembe (blue). In prototypical networks, each of the 5 support vectors are averaged to create a prototype for each class ($c_k$). The query vector $x$ is compared against each of the prototypes using squared eucldean distance. The query vector (shown as $x$) is assigned to the class of the prototype that it is most similar to. Here, the prototypes $c_k$ are shown as black circles. 
```

Prorotypical networks {cite}`snell2017prototypical` work by creating a single embedding vector  for each class in the support set, called the **prototype**. The prototype for a class is the mean of the embeddings of all the examples in the support set for that class.

The prototype (denoted as $c_k$) for a class $k$ is defined as:

$$
c_k = 1 / |S_k| \sum_{x_k \in S_k} f_\theta(x_k)
$$

where $S_k$ is the set of all examples in the support set that belong to class $k$, $x_k$ is an example in $S_k$, and $f_\theta$ is the neural net model we are trying to learn. 

After creating a prototype for each class in the support set, we use the euclidean distance between the query example and each prototype to determine which class the query example belongs to. We can build a probability distribution over the classes by applying a softmax function to the negated distances between a given query example and each prototype:

$$
p(y = k | x_q) = \frac{exp(-d(x_q, c_k))}{\sum_{k'} exp(-d(x_q, c_{k'}))}
$$

where $x_q$ is a query example, $c_k$ is the prototype for class $k$, and $d$ is the squared euclidean distance between two vectors.

### Prototypical Networks are Zero-Shot Learners too!
% TODO

## Optimization-based approaches

Optimization-based approaches focus on learning model parameters $\theta$ that can easily adapt to new tasks, and thus new classes. A canonical example of optimization-based few-shot learning is Model-Agnostic Meta Learning (MAML) {cite}`finn2017model`,
and it's successors {cite}`TODO`. 

The intuition behind MAML is that some representations are more easily transferrable to new tasks than others. 

For example, if we train a model to classify between `piano` and `guitar` audio samples, the model will have learned some parameters $\theta$ that are useful for classifying between `piano` and `guitar` audio samples. Normally, we would expect that these parameters $\theta$ would not be useful for classifying between instruments outside the training distribution, like `cello` and `flute`. The goal of MAML is to be able to learn parameters $\theta$ that are useful for classifying between `piano` and `guitar` audio samples, but will be easy to adapt to new instrument classification tasks given a support set for each task, like `cello` vs `flute`, `violin` vs `trumpet`, etc.

In other words, if we have some model parameters $\theta$, we want $\theta$ to be adapted to new tasks using only a few labeled examples (a single support set) in a few gradient steps. 

The MAML algorithm accomplishes this by training the model to adapt from a starting set of parameters, $\theta$, to a new set of parameters $\theta_i$ that are useful for a particular episode $E_i$. This is performed for all episodes in a batch, eventually learning a starting set of parameters $\theta$ that can be successfully adapted to new tasks using only a few labeled examples.

Note that MAML makes no assumption of the model architecture, thus the "model-agnostic" part of the method.

### The MAML algorithm

Suppose we are given a meta-training set composed of many few-shot episodes $D_{train} = \{E_1, E_2, ..., E_n\}$, where each episode contains a support set and train set $E_i = (S_i, Q_i)$. We can follow the MAML algorithm to learn parameters $\theta$ that can be adapted to new tasks using only a few examples, and a few gradient steps. 


Overview of the MAML algorithm {cite}`finn2017model`:
1. Initialize model parameters $\theta$ randomly, choose a step sizes $\alpha$ and $\beta$.  
2. **while** not converged **do**

    3. Sample a batch of episodes (tasks) from the training set $D_{train} = \{E_1, E_2, ..., E_n\}$
    4. **for** each episode $E_i$ in the batch **do**

        5. Using the current parameters $\theta$, compute the gradient $\nabla_{\theta} L_i f(\theta)$ of the loss $L_if(\theta)$ for episode $E_i$.

        6. Compute a new set of parameters $\theta_i$ by fine-tuning in the direction of the gradient w.r.t. the starting parameters $\theta$: 
        $$\theta_i = \theta - \alpha \nabla_{\theta} L_i$$

    7. Using the fine-tuned parameters $\theta_i$ for each episode, make a prediction and compute the loss $L_{i}f(\theta_i)$.

    8. Update the starting parameters $\theta$ by taking a gradient step in the direction of the loss we computed with the fine-tuned parameters $L_{i}f(\theta_i)$:
    $$\theta = \theta - \beta \nabla_{\theta} \sum_{E_i \in D_{train}}L_i f(\theta_i)$$


At inference time, we are given a few-shot learning task with support and query set $E_{test} = (S_{test}, Q_{test})$. We can use the learned parameters $\theta$ as a starting point, and follow a process similar to the one above to make a prediction for the query set $Q_{test}$:  

1. Initialize model parameters $\theta$ to the learned parameters from meta-training.
2. Compute the gradient $\nabla_{\theta} L_{test} f(\theta)$ of the loss $L_{test}f(\theta)$ for the test episode $E_{test}$.
3. Similar to step 6 of the training algorithm above, compute a new set of parameters $\theta_{test}$ by fine-tuning in the direction of the gradient w.r.t. the starting parameters $\theta$. 
4. Make a prediction using the fine-tuned parameters $\theta_{test}$: $\hat{y} =(\theta_{test})$.

## Memory-based approaches

TODO
