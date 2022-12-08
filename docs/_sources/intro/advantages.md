## Advantages of FSL and ZSL in MIR

Scientifically, FSL and ZSL are steps towards understanding how to achieve more human-like artificial intelligence as they are inspired by human ability. Humans are good at recognizing categories with very little direct supervision. For example, we can easily generalize the concept of the *"piano sound”* from a single audio clip rather than a million clips a deep learning model would need. 

Other than the scientific importance, FSL and ZSL also show some important advantages that could be particularly useful for MIR research: 

#### 1. Learning *rare*, *fine-grained*, and *novel* classes
FSL and ZSL techniques show the most significant advantages over other learning paradigms in learning classes where large-scale data collection is hard or simply impossible. For example, a rare instrument that only appears in a specific music genre or a newly designed sound effect that did not even exist before a model was trained. With FSL and ZSL, we are no longer limited to a pre-defined and fixed set of classes that we have enough labeled data. Instead, we can generalize to any class of interest at the cost of a little human intervention. For example, a few-shot drum transcription model can transcribe a new percussion instrument, unseen during training, by asking only for five audio examples of the target instrument from the user {cite}`wang2020fewshotdrum`. 

#### 2. Open up new possibilities for human-machine interaction
FSL and ZSL models naturally incorporate human input at test time without retraining the model or asking for significant human effort. This makes them useful tools when developing interactive MIR systems that can be customized on-the-fly by an individual user. FSL and ZSL models can adapt to the different needs of different users (e.g. different classes of interest), opening up new possibilities in human-machine interaction in MIR.

