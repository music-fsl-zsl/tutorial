## What is Few-Shot Learning (FSL) and Zero-Shot Learning (ZSL)?

Before dive right into FSL and ZSL, we would like to start from a brief discussion about the labeled data scarcity issue in MIR to show the motivation and relevance.  

### Labeled data scarcity issue in MIR
Deep learning has been highly successful in data-intensive applications, but is often hampered when the data set is small. A deep model that generalizes well typically needs to be trained on a large amount of labeled data. However, most MIR datasets are small in size compared to datasets in other domains such as image and text. This is not only because collecting musical data may run into copyright issues, but annotating them can also be very costly. The annotation process often requires expert knowledge and takes a long time as we need to listen to audio recordings multiple times. Therefore, many current MIR studies are built upon relatively small datasets with less ideal model generalizability. MIR researchers have been studying strategies to tackle this labeled data scarcity issue. These strategies can be roughly summarized into two catergories:

- Data: crowdsourcing, data augmentation, data synthesis
- Learning parasigm: transfer learning, unsupervised learning, semi-supervised learning

However, these methods either still go through large-scale data collection, having issues generalizing to real-world audio, or requiring a significant amount of labeled data (e.g. hundreds to thousands) for the target downstream tasks which could still be hard for rare classes. 


### FSL and ZSL
Few-shot learning (FSL) and zero-shot learning (ZSL), on the other hand, tackling the labeled data scarcity issue from a different angle. They are learning paradigms that aim to learn a model that can learn a new concept (e.g. recognizing a new class) quickly, based on just *a handful of labeled examples* (few-shot) or some *side information* (zero-shot). 


### An example
To have a better idea of how FSL and ZSL differ from standard supervised learning, let's consider an example of training a musical instrument classifier. We assume that there is an existing dataset in which we have abundant labeled examples for common instruments.

- In a standard supervised learning scenario, we train the classification model on the training set, then at test time, the model classifies *"new examples"* of *"seen classes"*.
- In a few-shot learning scenario, we also train the model with a same training set that is available. But at test time, the goal is for the few-shot model to recognize *"new examples"* from *"unseen classes"* by providing a very small amount of labeled examples.  
- In a zero-shot learning scenario, 

![Alt text](/assets/supervised_vs_fsl_vs_zsl.png)
