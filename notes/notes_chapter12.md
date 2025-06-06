# Chapter 12: Parallelizing Neural Network Training with Pytorch

## Summary what I learned
In this chapter, I have learned:
- What PyTorch is and how to use PyTorch to manipulate tensors.
- How to build input pipilnes in PyTorch.
- Use the `torch.nn` module to build complex machine learning and NN models and run them efficiently.
- Activation functions tanh, softmax, and ReLU.

## Learn some new classes, methods...
- Mathematical operations: similar to NumPy.
- `torch.chunk()`: devides and input tensor into a list of equally sized tensors.
- `torch.split()`: we can specify the size of output tensors directly instead of defining the number of splits.
- `torch.stack()`: cancatenates tensors along a new demension.
- `torch.cat()`: concatenates tensors in the given dimension.
- `torch.utils.data.DataLoader`: represents a Python iterable over a dataset.
- `torch.utils.data.Dataset`: an abstract class representing a dataset.
- `torch.utils.data.TensorDataset`: dataset wraping tensors.
- `pathlib.Path()`: represents concrete paths
- `pathlib.Path.glob()`: glob the given relative pattern in the directory, yielding all matching files.
- `torchvision.transforms.Compose()`: composes several transforms together.
- `itertools.islice()`: makes an iterator that return selected elements from iterable
- `torch.nn`: a module that contains classes and functions to build NNs.
- `torch.nn.MSELoss()`: creates a criterion that measures mean squared error between each element in the input x and target y.
- `torch.nn.Linear()`: applies an affine linear transformation to the incoming data
- `torch.optim.SGD()`: implements stochastic gradient descent.
- `Opimizer.zero_grad()`: reset the gradents of all optimized tensor.
- `torch.nn.CrossEntropyLoss()`: compute cross entropy loss.
- `torch.optim.Adam()`: implements adam algorithm.
- `torch.save()`: saves an object to a disk file.
- `torch.load()`: loads an object saved with `torch.save()` from a file.
- And some activation functions.

Create a `Dataset` class, define new method: `__init__()`, `__len__()`, `__getitem__()`.


## Explore some datasets from `torchvision`
- CelebA dataset
- MNIST dataset


## Code practice
In this chapter, I have created 3 playground notebooks:
1.  [practice_ch12.ipynb](../practice/practice_ch12.ipynb)
    - Load CelebA dataset
    - Explore dataset's target: attributes. Make a simple classification problem for the 20th attr: 'Male'
    - Build transform pipeline: resize, grayscale, totensor, normalize
    - Compute `male_indices`, `female_indices`, random select 1000 examples from each class.
    - Build `ImageDataset` class, return transformed images, label of the 20th attr.
    - Use `DataLoader` class to load data by batches and shuffled order.
    - Build a simple NN model using `nn.Linear` and `relu`.
    - Train model with `nn.CrossEntropyLoss` and `torch.optim.Adam`
    - Show images of wrong classified example.

2.  [practice_ch12_1.ipynb](../practice/practice_ch12_1.ipynb)
    This notebook is a more completely version of the previous practice file.
    - Evaluate model in each epoch, print Train loss, Train accuracy, Valid loss, Valid accuracy
    - Plot train/valid loss curve, train/valid accuracy curve.

3.  [practice_ch12_2.ipynb](../practice/practice_ch12_2.ipynb)
    In this notebook, I practice the techniques I have learned, but on the MNIST dataset.
    - Build a more complete model with `eval()` and `train()` methods.
    - show images of wrong classified examples.