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