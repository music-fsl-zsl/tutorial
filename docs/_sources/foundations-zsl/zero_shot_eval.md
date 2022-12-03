# Zero-shot learning evlauation

There are different ways for setting up the zero-shot inference situation. 


### Seen / unseen label split 

- Inductive zero-shot learning 
    - Only labeled training samples and auxiliary information of seen classes are available during training.
- Semantic transductive zero-shot learning 
    - Labeled training samples and auxiliary information of all classes are available during training.
- Transductive zero-shot learning 
    - Labeled training samples, unlabelled test samples, and auxiliary information of all classes are available during training.


<img src = "../assets/zsl/data_split.png" width=600>


### 'Generalized' zero-shot evaluation setup

- Generalized evaluation scheme. 
<p align = "left">
<img src = "../assets/zsl/gzsl_vs_zsl.png" width=600>
</p>
<p align = "left">
A schematic diagram of ZSL versus GZSL. Assume that the seen class contains samples of Otter and Tiger, while the unseen class contains samples of Polar bear and Zebra. (a) During the training phase, both GZSL and ZSL methods have access to the samples and semantic representations of the seen class. (b) During the test phase, ZSL can only recognize samples from the unseen class, while (c) GZSL is able to recognize samples from both seen and unseen classes.
</p>

### Tricky business with multi-label zero-shot evaluation

- <img src = "../assets/zsl/zsl_music_split.png" width=400> 


### Evlauation metrics

- AUCROC
- Per-class Top-1 accuracy 
    - ensures that all classes will weigh the same
$$
acc_{\mathcal{Y}}=\frac{1}{\|\mathcal{Y}\|} \sum_{c=1}^{\|\mathcal{Y}\|} \frac{\text {number of correct predictions in class c}}{\text {number of instances in class c}}
$$
- Harmonic Mean for GZSL
    - ensures that seen and unseen class accuracy will weigh the same
$$
H=\frac{2 * acc_{\mathcal{Y}^{seen}} * acc_{\mathcal{Y}^{unseen}}}{acc_{\mathcal{Y}^{seen}}+acc_{\mathcal{Y}^{unseen}}}
$$

