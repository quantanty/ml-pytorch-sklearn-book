# Chapter 14: Classifying Image with Deep Convolutional Neural Networks

## CNNs
- CNNs were originally inspired by how the visual cortex of the human brain works when recognizing objects. They are often described as "feature extraction layers".
- extracting salient features.
- automatically learn the features from raw data.
### Understanding CNNs and Feature hierarchies:
- low-level features are combined into high-level features.
- local receptive field: local patch of pixels.
- CNNs usually perform very well on image-related tasks, due to two important ideas:
    - Sparse conectivity
    - Parameter sharing
- pooling layers: do not have any learnable parameters.
- Typically, CNNs are composed of several convolution layers and subsampling layers that are followed by fully connected layers at the end.

### Discrete convolutions
Discrete convolution in one dimension:

$$ \large
\mathbf{y} = \mathbf{x} * \mathbf{w} \rightarrow y[i] = \sum_{k=-\infty}^{+\infty} x[i-k] w[k]
$$

Cross-correlation:

$$ \large
\mathbf{y} = \mathbf{x} * \mathbf{w} \rightarrow y[i] = \sum_{k=-\infty}^{+\infty} x[i+k] w[k]
$$

Hyper parameters: padding, stride.
3 modes of padding commonly used: full, same, valid.

**Should** perserve the spatial size using same padding for the conv. layers and decrease the spatial size via pooling layers or conv. layers with stride 2 instead.

Convolution output size:

$$ \large
o = \Bigg\lfloor \frac{n + 2p - m}{s} \Bigg\rfloor + 1
$$

Discrete convolution in 2D:

$$ \large
\mathbf{Y} = \mathbf{X} * \mathbf{W} \rightarrow Y[i, j] = \sum_{k_1=-\infty}^{+\infty} \sum_{k_2=-\infty}^{+\infty} X[i-k_1, j-k_2] W[k_1, k_2]
$$

### Subsampling layers
2 forms of pooling operations: max-pooling and mean-pooling.

Advantage:
- local invariance $\to$ generate features that are more robust to noise in the input data.
- decrease size of features $\to$ $\uparrow$ computational efficiency, $\downarrow$ overfitting.

Non-overlapping pooling: stride equal to pooling size. Traditional.
Overlapping pooling: stride smaller than pooling size.

## Implementing a CNN
### Working with multiple input channels, output channels
Given an example $\mathbf{X}_{n_1 \times n_2 \times C_{in}}$, a kernel matrix $\mathbf{W}_{m_1 \times m_2 \times C_{in} \times C_{out}}$, and a bias vector $\mathbf{b}_{C_{out}}$

$$
\implies
\begin{cases}
\bm{Z}^{\text{conv}}[:,:,k] = \sum_{c=1}^{C_{in}} \bm{W}[:,:,c,k] * \bm{X}[:,:,c] \\
\text{Pre-activation:} \quad \bm{Z}[:,:,k] = \bm{Z}^{\text{conv}} + b[k] \\
\text{Feature map:} \quad \bm{A}[:,:,k] = \sigma(\bm{Z}[:,:,k])
\end{cases}
$$

### Regularizing an NN with L2 regularization and dropout
In real world machine learning problems, we do **not know** how large the network should be a priori.
$\rightarrow$ We should build a network with a **relatively large** capacity (slightly larger than necessary) to do well on the training set. Then, to prevent overfitting, we can apply one or multiple **regularization** schemes to achieve good generalization performance on new data.

Though L2 regularization and `weight_decay` are not strictly identical, they are equivalent when using SGD optimizers.

**Dropout**: regularize deep NNs, avoid overfitting, improving the generalization performance.
Effect: force the network to learn a redundant representation of the data, force the network to learn more general and robust patterns from the data.
During prediction: all neurons will contribute to computing the pre-activation of the next layer.

### Loss function for classification
Binary classification:
- `nn.BCEWithLogitsLoss()` $\leftarrow$ logits
- `nn.BCELoss()` $\leftarrow$ probabilites

Multiclass classification:
- `nn.CrossEntropyLoss()` $\leftarrow$ logits
- `nn.NNLLoss()` $\leftarrow$ probabilities

### Other techniques

**Data augmentation**: $\uparrow$ generalization performance, $\downarrow$ overfitting on small dataset.

**Global average-pooling**: calculates the average of each channel, decrease number of features, avoid overfitting.

## Related Articles
- *Handwritten Digit Recognition with a Back-Propagation Network* by *Y. LeCun, and colleagues*, 1989.

- *Striving for Simplicity: The All Convolutional Net ICLR (workshop track)*, by *Jost Tobias Springenberg, Alexey Dosovitskiy*, and others, 2015.

- *Fast Algorithms for Convolutional Neural Networks* by Andrew Lavin and Scott Gray, 2015.

- *ImageNet Classification with Deep Convolutional Neural Networks* by *A. Krizhevsky, I. Sutskever, and G. Hinton*, 2012.

- *Striving for Simplicity: The All Convolutional Net* by *Jost Tobias Springenberg, Alexey Dosovitskiy, Thomas Brox*, and *Martin Riedmiller*, 2014.

- *VoxNet: A 3D Convolutional Neural Network for Real-Time Object Recognition* by *Daniel Maturana* and *Sebastian Scherer*, 2015.

- *Decoupled Weight Decay Regularization* by *Ilya Loshchilov* and *Frank Hutter*, 2019.

- *Dropout: A Simple Way to Prevent Neural Networks from Overfitting* by *N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever*, and *R. Salakhutdinov*, *Journal of Machine Learning Research 15.1*, pages 1929-1958, 2014

- *Adam: A Method for Stochastic Optimization* by *Diederik P. Kingma* and *Jimmy Lei Ba*, 2014