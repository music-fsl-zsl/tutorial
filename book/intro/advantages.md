## Advantages of FSL and ZSL in MIR

Scientifically, FSL and ZSL are steps towards understanding how to achieve more human-like artificial intelligence as they are inspired by human ability. Humans are good at recognizing categories with very little direct supervision. For example, we can easily generalize the concept of *"the sound of piano”* from a single audio clip rather than a million clips a deep learning model would need. 

Other than this scientic importance, FSL and ZSL also show some important advantages that could be particularly useful for MIR research: 

#### 1. Learning *rare*, *fine-grained*, and *novel* classes
FSL and ZSL techniques show most significant advantages over other learning paradigms in learning classes where large-scale data collection is hard or simply impossible. For example, rare instruments or new sound effects. This is closely related to the challenges in many MIR research. Many MIR datasets are small in size compared to datasets in other domains such as image and text. This is not only because collecting musical data may run into copyright issues, but annotating them can also be very costly. The annotation process often requires expert knowledge and takes a long time as we need to listen to audio recordings multiple times. Therefore, many current MIR studies are built upon relatively small datasets with less ideal model generalizability or limited class vocabulary. With FSL and ZSL, we are no longer limited to a pre-defined and fixed set of classes but ideally can generalize to any class of interest with the cost of little human intervention. For example, a few-shot drum transcription model can transcribe a new percussion instrument, unseen during training, by asking only five audio examples of the target instrument from the user. 

#### 2. Open up new possibilities for human-machine interaction
FSL and ZSL models naturally incorporate human input at test time without retraining the model or asking significant human effort. This makes them useful tools when developing interactive MIR tools that can be customized by individual user on-the-fly. Instead of being limited by a fixed set of classes defined at training, FSL and ZSL models can adapt to different needs from different users, opening up new possibilities in human-machine interaction in MIR.

