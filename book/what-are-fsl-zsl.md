## What are Few-Shot Learning and Zero-Shot Leanring?

Deep learning has been highly successful in data-intensive applications, but is often hampered when the data set is small. 
Few-shot learning (FSL) and zero-shot learning (ZSL) are learning paradigms that aim to tackle this problem 
by training a model that can learn a new concept (e.g. recognizing a new class) based on just a handful of labeled examples (few-shot) or some auxiliary information (zero-shot). 
This is inspired by human learning ability. Humans are good at recognizing categories with very little direct supervision. 
For example, children can easily generalize the concept of “cat” from a single picture rather than a million pictures a deep learning model would need.

FSL and ZSL techniques show most significant advantages in learning rare, fine-grained, or novel classes 
where large-scale data collection is hard or simply impossible.
This is particulary relevent to many MIR research given that many MIR datasets are small in size compared to datasets in other domains such as image and text. 
This is not only because collecting musical data may run into copyright issues, 
but annotating them can also be very costly as it often requires expert knowledge. 

In addition, few-shot and zero-shot models naturally incorporate human input without asking significant human effort, 
making them useful tools when developing MIR systems that can be customized by individual users.
