# Music Source Separation

Music source separation(MSS) is a well-studied problem with various applications, where the goal is to separate the sound of particular instruments from a mixture recording.

## Limitations of existing MSS models
Like in many other fields, deep learning-based approaches have achieved promising results for MSS. These models are typically trained to separate a particular instrument class {cite}`jansson2017singing`. So we need multiple models for multi-instrument separation. As the number of instruments increases, we soon run into scaling issues.

To overcome this limitation, more recent approaches use a single generic model with an instrument class conditioning mechanism {cite}`slizovskaia2019end`{cite}`meseguer2019conditioned`. Where we can use side information, here, the instrument class label, to configure the model to separate different instruments. However, these models only support a fixed number of instruments that the models were trained on, and do not generalize to unseen instruments. This greatly limits the applicability of MSS in real-world situations involving a larger and growing scope of instrument classes. 
 
## Few-shot MSS
To address this issue, Wang et al. {cite}`wang2022fewshot` propose a few-shot MSS approach. Instead of specifying the class label, the source separation model is directly conditioned on a few audio examples of the target instrument. With this paradigm, we can separate any instrument of interest as long as we have a few examples of it.

The proposed few-shot MSS framework consists of three main components.
1. **A standard U-Net source separation model** {cite}`meseguer2019conditioned`- encoding the input mixture into a bottleneck feature, and decoded it to get the separated audio. 
2. **A few-shot conditioning encoder** - embedding the audio examples of the target instrument, and taking the average to get a single conditioning vector.
3. **FiLM conditioning mechanism** {cite}`perez2018` - transforming the bottleneck feature in U-Net by scaling and shifting it based on the conditioning vector. 

```{figure} ../assets/advances/few-shot-mss.png
---
name: few-shot-mss
---
```

The results show that the few-shot MSS model, conditioned on 5 audio examples of the target instrument, outperforms the instrument class conditioned baseline on both *seen* and *unseen* instruments, and shows more significant advantage on unseen instruments.

## Negative conditioning examples
Another interesting idea proposed in this work is leveraging *negative* conditioning information. The basic few-shot MSS model is extended to incorporate additional examples with *unwanted* instrument into the final conditoining vector. The result shows that providing additional information about what not to separate further improves the separation. This setup can be particularly useful when labeling non-target instruments is much easier than labeling the target one. 

```{figure} ../assets/advances/few-shot-mss-neg.png
---
name: few-shot-mss-neg
---
```

