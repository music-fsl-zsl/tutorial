# Approaches

Now that we have a grasp of the foundations of few-shot learning, 
we'll take a look at some of the most common approaches to the solving few-shot problems. 

Recall that the goal of few-shot learning is to be able to learn to solve a new machine learning task given only a few labeled examples. In few-shot learning problems, we are given a small set of labeled examples for each class we would like to predict (the support set), as well as a larger set of unlabeled examples (the query set). We tend to refer to few-shot learning tasks as $K$-way, $N$-shot classification tasks, where $K$ is the number of classes we would like to predict, and $K$ is the number of labeled examples we are given for each class. 

When training a model to solve a few-shot learning task, we typically sample episodes from a large training dataset. An episode is a simulation of a few-shot learning task, where we sample $K$ classes and $N$ labeled examples for each class. Training a deep model by sampling few-shot learning episodes from a large training dataset is known as **episodic training**.

Here are the few-shot learning approaches covered in this tutorial:
1. [Metric-based few-shot learning](metric-based-fsl.md)

2. [Optimization-based few-shot learning](optimization-based-fsl.md)

3. [Memory-based few-shot learning](memory-based-fsl.md)