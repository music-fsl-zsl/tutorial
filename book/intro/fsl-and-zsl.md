## What are Few-Shot Learning(FSL) and Zero-Shot Learning(ZSL)?

### Labeled data scarcity issue in MIR
Deep learning has been highly successful in data-intensive applications, but is often hampered when the data set is small. A deep model that generalizes well typically needs to be trained on a large amount of labeled data. However, most MIR datasets are small in size compared to datasets in other domains such as image and text. This is not only because collecting musical data may run into copyright issues, but annotating them can also be very costly. The annotation process often requires expert knowledge and takes a long time as we need to listen to audio recordings multiple times. Therefore, many current MIR studies are built upon relatively small datasets with less ideal model generalizability. MIR researchers have been studying strategies to tackle this labeled data scarcity issue including data augmentation, data synthesizing, crowd-sourcing, transfer learning, unsupervised learning, etc. These approaches typically either collect, generate, or leverage **more data** to train larger and more powerful models to generalize to more use cases.

### FSL and ZSL
Few-shot learning (FSL) and zero-shot learning (ZSL), on the other hand, are learning paradigms tackling this problem that aims to learn a model that can learn a new concept (e.g. recognizing a new class) based on just *a handful of labeled examples* (few-shot) or some *auxiliary information* (zero-shot). 

```{note}
Note: There exists different definitions of the scope of FSL and ZSL. For example, data augmentation and transfer learning can sometimes be categorized as FSL strategies. In this tutorial, we consider a more confined definition as mentioned above. 
```

### An example
To have a better idea of how FSL and ZSL different from standard supervised learning, let's consider an example of training a musical instrument classifier. We assume that there is an existing dataset in which we have abundant labeled examples for common instruments.

- In a standard supervised learning scenario, we train the classification model on the training set, then at test time, the model classifies *"new examples"* of *"seen classes"*.
- In a few-shot learning scenario, we also train the model with a same training set that is available. But at test time, the goal is for the few-shot model to recognize *"new examples"* from *"unseen classes"* by providing a very small amount of labeled examples.  
- In a zero-shot learning scenario, [TODO]

![Alt text](/assets/supervised_vs_fsl_vs_zsl.png)
