# NMT-with-Attention

## What

In this repository, I use s seq2seq model which internally uses an Encoder-Decoder network with attention mechanism to achieve the task of Neural Machine Translation (NMT). Specifically, we try to do Spanish to English translation.   
The key idea of the attention mechanism is to establish direct short-cut connections between the target and the source by paying "attention" to relevant source content as we translate. A nice byproduct of the attention mechanism is an easy-to-visualize alignment matrix (annotation weights) between the source and target sentences

## Why

To imporve upon vanilla seq2seq models as for long sentences, the single fixed-size hidden state becomes an information bottleneck. So we use attention mechanism as it improves the translation of longer sentences. We look at the internal details of attention mechanisms by using the Subclassing API provided by Tensorflow. And also visualize the annotation weights which tell us which positions in the source sentence were considered more important when generating the target word.

## How

We have used an Encoder-Decoder network with Bahdanau attention. During training, we use Teacher Forcing. The following figure roughly illustrates our model.

<img src="Images/attention.jpg" alt="Attention" width="400"/>   

As can be seen in the figure, attention is calculated at every time step. It consists of the following steps:

1. The current target hidden state is compared with all source states to derive *attention weights*
2. Based on the attention weights we compute a *context vector* as the weighted average of the source states
3. Combine the context vector with the current target hidden state to yield the final *attention vector*
4. The attention vector is fed as an input to the next time step (*input feeding*)

During inference, no Teacher Forcing is used. Rather, the input to the decoder at each time step is its previous predictions along with the hidden state and the encoder output. We also plot the alignment matrix for each translation.
## Conclusion

We can see from the plots of annotation weights that the alignment of words between Spanish and English is largely monotonic. We see strong weights along the diagonal of each matrix. However, we also observe a number of non-trivial, non-monotonic alignments. Also, since our model was trained only on a subset of training data, we may get a slighlty wrong translation as in the last case.
