# Drum Transcription

Building on top of the sucess of few-shot classification among monophonic audio, Wang et al. applied FSL to polyphonic musical audio to perfom few-shot drum transcription.  

Automatic Drum Transcription (ADT) aims at deriving a symbolic annotation of percussion instrument events from a music recording. An accurate ADT system enables diverse applications in music education, music production, music search and recommendation, and computational musicology. Again, data-driven approaches to ADT are often limited to a predefined, small vocabulary (3-20) of percussion instrument classes due to the limited number and size of ADT datasets, and the small class vocabulary size of the annotations in these datasets. Such models cannot recognize out-of-vocabulary classes nor are they able to adapt to finer-grained vocabularies. 

In this work, the authors address open vocabulary ADT by introducing FSL to the task. They train a Prototypical Network on a synthetic dataset and evaluate the model on multiple real-world ADT datasets with polyphonic accompaniment. We show that, given just a handful of selected examples at inference time, they can match and in some cases outperform a state-of-the-art supervised ADT approach under a fixed vocabulary setting. At the same time, they show that few-shot model can successfully generalize to finer-grained or extended vocabularies unseen during training, a scenario where supervised approaches cannot operate at all.

