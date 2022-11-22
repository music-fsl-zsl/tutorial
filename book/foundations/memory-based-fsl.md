# Memory Based Few-Shot Learning

A third approach to few-shot learning looks to extract knowledge from a large training dataset and store it some kind of external memory. 
During inference, the model can leverage this external memory to make predictions on novel classes. 

This approach is known as **memory based few-shot learning**.

Memory-based approaches make use of an external key-value memory module, $M$, which stores some kind of representation of the training data. 
The memory module is typically some matrix $M \in \mathbb{R}^{b \times m}$, which can be thought of as a 
lookup table with $b$ memory slots. Each memory slot $M(i)$ is a vector of length $m$.

During inference, given some query example $x_q$, we can embed the query example into the learned embedding space using a neural network $f_\theta$. 
We can make use of this embedding $f_\theta(x_q)$ to find the memory slots in $M$ that are most similar to the query example. 
We use a similarity metric $g_{sim}$ to compare the query embedding to each key in $M$: $g_{sim}(f_\theta(x_q), M_key(i))$. 

After finding the most similar memory keys, we can use the values in those memory slots to make predictions. 

To illustrate an approach to memory based few-shot learning, we'll talk about memory-augmented neural networks (MANNs) {cite}`santoro2016meta`.



