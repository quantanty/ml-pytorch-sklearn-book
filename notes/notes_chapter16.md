# Chapter 16: Transformers - Improving Natural Language Processing with Attention Mechanisms

---

## Topics covered in this chapter
- Improving RNNs with an attention mechanism
- Introducing the stand-alone self-attention mechanism
- Understanding the original transformer architecture
- Comparing transformer-based large-scale language models
- Fine-tuning BERT for sentiment classification

---

## New functions, classes I learned
- `torch.allclose()`
- `transformers.pipeline()`
- `transformers.GPT2Tokenizer.from_pretrained()`
- `transformers.GPT2Model.from_pretrained()`
- `requests.get()`
- `gzip.open()`
- `shutil.copyfileobj()`
- `transformers.DistilBertTokenizerFast.from_pretrained()`
- `transformers.DistilBertForSequenceClassification.from_pretrained()`
- `transformers.Trainer`
- `transformers.TrainingArguments`

---

## Theoretical notes
### Multihead self-attention

$$ \large
q_i^c=x_iW^{Qc}; k_i^c=x_iW^{Kc}; v_i^c=x_iW^{Vc} \\
\text{score}^c(x_i, x_j) = \frac{q_i^c {k_i^c}^T}{\sqrt{d_k}} \\
\alpha_{ij}^c = \sum_{j=1}^T \alpha_{ij}v_j^c \\
z_i^{\text{multihead}} = \text{concat}(z_i^1, z_i^2,...,z_i^h) \\
z_i^{\text{output}} = z_iW^O+b^O
$$

$$ \large
\text{Attention}(Q,K,V) = \text{softmax}\Bigg(\frac{QK^T}{\sqrt{d_k}}\Bigg)V
$$

### Mask self-attention

$$ \large
\text{Attention}(Q,K,V,M) = \text{softmax}\Bigg(\frac{QK^T+M}{\sqrt{d_k}}\Bigg)V
$$

### Transformer blocks

$$ \large
\begin{align}
t_i^1 &= \text{LayerNorm}(x_i) \\
t_i^2 &= \text{MultiHeadAttention}(t_i^1, [t_1^1,...,t_N^1]) \\
t_i^3 &= t_i^2 + x_i \\
t_i^4 &= \text{LayerNorm}(t_i^3) \\
t_i^5 &= \text{FFN}(t_i^4) \\
h_i &= t_i^5 + t_i^3
\end{align}
$$

---

## Further reading
- Attention Is All You Need (A. Vaswani et al., 2017)
- Layer Normalization (J.L. Ba et al., 2016)
- Deep contextualized word representations (M.E. Peters et al., 2018)
- Improving Language Understanding by Generative Pre-Training (A. Radford et al., 2018)
- Language Models are Unsupervised Multitask Learners (A. Radford et al., 2019)
- Language Models are Few-Shot Learners (T.B. Brown et al., 2020)
- Generating Long Sequences with Sparse Transformers (R. Child et al., 2019)
- BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (J. Devlin et al., 2018)
- BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension (M. Lewis et al., 2019)
- Pre-trained Models for Natural Language Processing: A Survey (X. Qiu et al., 2020)
- AMMUS : A Survey of Transformer-based Pretrained Models in Natural Language Processing (K.S. Kalyan et al., 2021)

---

## Challenges encountered
- My English skills are still limited. I read slowly, so I need to practice more to improve.
- Various documents employ different notations, and it took me some time to become accustomed to each of them.
- These language models are often very large, so fine-tuning also takes a long time. I find it difficult to adjust parameters to produce the best model. Instead, I spend my time practicing by understanding the pipeline.

---

## Advanced topics & future directions
- Prompt Engineering
- Retrieval-Augmented Generation (RAG)
- Mixture of Expert (MoE)
- Multimodal