## What are Few-Shot Learning(FSL) and Zero-Shot Leanring(ZSL)?

Deep learning has been highly successful in data-intensive applications, but is often hampered when the data set is small. 
Few-shot learning (FSL) and zero-shot learning (ZSL) are learning paradigms tackling this problem by training a model that can learn a new concept (e.g. recognizing a new class) based on just a handful of labeled examples (few-shot) or some auxiliary information (zero-shot). 


Considering an example of training a handwritten digit classifier. In a standard supervised learning scenario, we train the classification model on a large amount of training data for each class of interest. Then at test time, the model classifies *"new examples"* of *"seen classes"*. On the other hand, in a few-shot learning scenario, we also train the model with a large training set that is available. But at test time, the goal is for the few-shot model to recognize *"new examples"* from *"unseen classes"* by providing a very small amount of labeled examples.  

![Alt text](./supervised_vs_fsl_vs_zsl.png?raw=true "Title")


## Why do we care about FSL and ZSL?

Scientifically, FSL and ZSL are steps towards understanding how to achieve more human-like artificial intelligence as they are inspired by human ability. Humans are good at recognizing categories with very little direct supervision. For example, children can easily generalize the concept of *"the sound of piano”* from a single audio clip rather than a million clips a deep learning model would need.

Application wise, the main advantages of FSL and ZSL are:

#### 1. Reduce annotation effort
FSL and ZSL techniques show most significant advantages in learning rare, fine-grained, or novel classes where large-scale data collection is hard or simply impossible, which is closely related to the challenges in many MIR research. Many MIR datasets are small in size compared to datasets in other domains such as image and text. This is not only because collecting musical data may run into copyright issues, but annotating them can also be very costly. The annotation process often requires expert knowledge and takes a long time as we need to listen to audio recordings multiple times. Therefore, many current MIR studies are built upon relatively small datasets with less ideal model generalizability or limited class vocabulary. With FSL and ZSL, we are no longer limited to a pre-defined and fixed set of classes but ideally can generalize to any class of interest with the cost of little human intervention. For example, a few-shot drum transcription model can transcribe a new percussion instrument, unseen during training, by asking only five audio examples of the target instrument from the user. 

#### 2. Open up new possibilities for human-machine interaction
FSL and ZSL models naturally incorporate human input at test time without asking significant human effort. This makes them useful tools when developing MIR systems that can be customized by individual users.

