# Modeling Sequential Data Using Recurrent Neural Networks

## Introducing sequential data
- elements in a sequence appear in a certain order and are not independent of each other. Not IID. order matters.
- e.g., predicting market value of a perticular stock.
- not all sequential data has the time dimension.
- represent: $x^{(1)}, x^{(2)}, ...,x^{(T)}$
- sequence modeling categories: many-to-one, one-to-many, many-to-many (input-output synchronized), many-to-many (input-output not synchronized)

## RNNs for modeling sequences
- at time step $t$, hidden state $h^{(t)}$ receives input $x$ and hidden state of the previous time step $h^{(t-1)}$

- The flow of information in adjacent time steps allows the network to have a memory of past events. It is usually displayed as a loop, also known as a **recurrent edge** in graph notation, which is how this general RNN arch. got its name.

- RNNs can consists of multiple RNN layers.

- Training RNNs using back propagation through time (BPTT)

- Hidden recurrence versus output recurrence

- The challenges of learning long-range interactions:
    - vanishing gradient
    - exploding gradient
    - long-range dependencies
    - solutions:
        - use LSTM or GRU
        - gradient clipping
        - Truncated backpropagation through time (TBPTT)

## Long short-term memory

## Implementing a multilayer RNN for sequence modeling in PyTorch.

## Project one: RNN sentiment analysis of the IMDb movie review dataset

## Project two: RNN character-level language modeling with LSTM cells, using text data from Jules Verne's *The Mysterious Island*

## Using gradicent clipping to avoid exploding gradients