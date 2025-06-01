
# Chapter 11: Implementing a Multilayer Artificial Neural Network from Scratch

## Some interesting papers/articles
- *A logical calculus of the ideas immanent in nervous activity, by W. S. McCulloch and W. Pitts, The Bulletin of Mathematical Biophysics*, 5(4):115–133, 1943.

- *Learning representations by backpropagating errors, by D.E. Rumelhart, G.E. Hinton, and R.J. Williams, Nature*, 323 (6088): 533–536, 1986.

- [AI winter - wikipedia](https://en.wikipedia.org/wiki/AI_winter)

- [Predicting COVID-19 resource needs from a series of X-rays](https://arxiv.org/abs/2101.04909)

- [Modeling virus mutations](https://science.sciencemag.org/content/371/6526/284)

- [Leveraging data from social media platforms to manage extreme weather events](https://onlinelibrary.wiley.com/doi/abs/10.1111/1468-5973.12311)

- [Improving photo descriptions for people who are blind or visually impaired](https://tech.fb.com/how-facebook-is-using-ai-to-improve-photo-descriptions-for-people-who-are-blind-or-visually-impaired/)

## Some theory notes

We can add any number of hidden layers to the MLP. Number of layers and units: hyperparameter.

Loss gradients will become increasingly small as more layers are added to a network. -> **Vanishing gradient** problem. -> Special algorithms have been developed to help train such DNN structures; this is known as **deep learning**.

Summarize MLP learning procedure in three simple steps:
1.  Starting at the input layer, forward propagate the patterns of the training data through the network to generate an output.
2.  Based on the network's output, calculate the loss that we want to minimize using a loss function.
3. Backpropagate the loss, find its derivative with respect to each weight and bias unit in the network, and update the model.