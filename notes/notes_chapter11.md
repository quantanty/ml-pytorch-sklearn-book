
# Chapter 11: Implementing a Multilayer Artificial Neural Network from Scratch


## Summary what I learned
In this chapter, I have learned:
- The basic concept behind multilayer artificial NNs.
- Connecting multiple neurons to a powerful NN architecture to solve complex problems.
- Backpropagation algorithm.

## Some theory notes

We can add any number of hidden layers to the MLP. Number of layers and units: hyperparameter.

Loss gradients will become increasingly small as more layers are added to a network. -> **Vanishing gradient** problem. -> Special algorithms have been developed to help train such DNN structures; this is known as **deep learning**.

Summarize MLP learning procedure in three simple steps:
1.  Starting at the input layer, forward propagate the patterns of the training data through the network to generate an output.
2.  Based on the network's output, calculate the loss that we want to minimize using a loss function.
3. Backpropagate the loss, find its derivative with respect to each weight and bias unit in the network, and update the model.


Partial derivtive of $\bm{W}$ with respect to each weight for every layer:
$$ \large
\frac{\partial L(\bm{W}, \bm{b})}{\partial w_{j, k}^{(l)}}
$$

Apply chain rule to compute gradients:
$$ \large
\frac{\partial L}{\partial \bm{Z}^{(out)}} = \frac{\partial L}{\partial \bm{A}^{(out)}}\frac{\partial \bm{A}^{(out)}}{\partial \bm{Z}^{(out)}}
$$

$$ \large
\frac{\partial L}{\partial \bm{A}^{(h)}} = \frac{\partial L}{\partial \bm{Z}^{(out)}}\frac{\partial \bm{Z}^{(out)}}{\partial \bm{A}^{(h)}}
$$

$$ \large
\frac{\partial L}{\partial \bm{Z}^{(h)}} = \frac{\partial L}{\partial \bm{A}^{(h)}}\frac{\partial \bm{A}^{(h)}}{\partial \bm{Z}^{(h)}}
$$

$$ \large
\frac{\partial L}{\partial w_{1,1}^{(out)}} = \frac{\partial L}{\partial \bm{Z}^{(out)}} \frac{\bm{Z}^{(out)}}{\partial w_{1,1}^{(out)}}
$$

## Some interesting papers/articles
- *A logical calculus of the ideas immanent in nervous activity, by W. S. McCulloch and W. Pitts, The Bulletin of Mathematical Biophysics*, 5(4):115–133, 1943.

- *Learning representations by backpropagating errors, by D.E. Rumelhart, G.E. Hinton, and R.J. Williams, Nature*, 323 (6088): 533–536, 1986.

- [AI winter - wikipedia](https://en.wikipedia.org/wiki/AI_winter)

- [Predicting COVID-19 resource needs from a series of X-rays](https://arxiv.org/abs/2101.04909)

- [Modeling virus mutations](https://science.sciencemag.org/content/371/6526/284)

- [Leveraging data from social media platforms to manage extreme weather events](https://onlinelibrary.wiley.com/doi/abs/10.1111/1468-5973.12311)

- [Improving photo descriptions for people who are blind or visually impaired](https://tech.fb.com/how-facebook-is-using-ai-to-improve-photo-descriptions-for-people-who-are-blind-or-visually-impaired/)

- *Deep residual learning for image recognition by K. He, X. Zhang, S. Ren, and J. Sun, Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp*. 770-778, 2016.

- *Cyclical learning rates for training neural networks by L.N. Smith, 2017 IEEE Winter Conference on Applications of Computer Vision (WACV)*, pp. 464-472, 2017.

- *Rethinking the Inception architecture for computer vision by C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna, Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, pp. 2818-2826, 2016.

- *Learning representations by backpropagating errors, by D.E. Rumelhart, G.E. Hinton, and R.J. Williams*, Nature, 323: 6088, pages 533–536, 1986.