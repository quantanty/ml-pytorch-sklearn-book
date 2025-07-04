# Chapter 17: Generative Adversarial Networks for Synthesizing New Data

---

## Topics covered in this chapter
- Introducing generative models for synthesizing new data
- Autoecoders, variational autoencoders, and their relationship to GANs
- Understanding the building blocks of GANs
- Implementing a simple GAN model to generate handwritten digits
- Understanding transposed convolution and batch normalization
- Improving GANs: deep convolutional GANs and GANs using the Wasserstain distance

---

## New functions, classes I learned
- `torch.nn.LeakyReLU`
- `np.prod()`
- `nn.ConvTranspose2d`
- `nn.BatchNorm2d`
- `nn.InstanceNorm2d`
- `torch.autograd.grad()`

---

## Further reading
- Generative Adversarial Nets (I. Goodfellow et al., 2014)
- Stacked Denoising Autoencoders: Learning Useful Representations in a Deep Network with a Local Denoising Criterion (P. Vincent et al., 2010)
- Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks (A. Radford et al., 2015)
- A guide to convolution arithmetic for deep learning (V. Dumoulin and F. Visin, 2016)
- Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift (S. Ioffe and C. Szegedy, 2015)
- How Does Batch Normalization Help Optimization? (Shibani Santurkar et al., 2018)
- Pros and Cons of GAN Evaluation Measures: New Developments (A. Borji, 2021)
- Wasserstein Generative Adversarial Networks (M. Arjovsky et al., 2017)
- Improved Training of Wasserstein GANs (I. Gulrajani et al., 2017)