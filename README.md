# GAIA Seminar: Introduction to Transformers

Interactive seminar on building a GPT-style transformer from scratch using PyTorch.

## 📚 Contents

- **`seminar_materials/`** - All seminar materials
  - `seminar_notebook_completed.ipynb` - Full interactive notebook with explanations, exercises, and solutions
  - `gpt_model.py` - Clean GPT model implementation
  - `utils_viz.py` - Visualization utilities
  - `vefxistyaosani.txt` - Georgian poetry dataset ([source](https://github.com/Anbani/anbani.db/blob/master/datasets/vefxistyaosani.txt))
  - `images/` - Diagrams and figures used in the notebook

## 🚀 Quick Start (Google Colab)

```python
# 1. Clone the repository
!git clone https://github.com/lukabarbakadze/GAIA-Seminar.git
%cd GAIA-Seminar/seminar_materials

# 2. Install dependencies
!pip install torch matplotlib seaborn scikit-learn

# 3. Open seminar_notebook_completed.ipynb and follow along
```

## 📖 What You'll Learn

1. **Character-level tokenization** for Georgian text (ვეფხისტყაოსანი)
2. **Self-attention mechanism** from first principles
3. **Transformer architecture** - building blocks step by step
4. **Training a GPT model** on Georgian poetry
5. **Text generation** with temperature sampling

## 🎯 Key Features

- **No prerequisites**: Builds from raw matrix operations to full transformer
- **Visual learning**: Integrated plots for attention, embeddings, training curves
- **Hands-on**: Train a real model that generates Georgian text
- **Modular**: Clean separation of model, utils, and teaching materials

## 🛠️ Requirements

```bash
torch
matplotlib
seaborn
scikit-learn
```

## 📖 Sources & References

### Essential Resources

- [LLM Visualization](https://bbycroft.net/llm) — Interactive 3D visualization of LLM internals
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) — Harvard NLP's line-by-line annotation of *Attention Is All You Need*
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Jay Alammar's visual guide to the Transformer architecture
- [Transformer Architecture (Towards AI)](https://pub.towardsai.net/transformer-architecture-part-1-d157b54315e6) — Detailed walkthrough of Transformer internals
- [Language Understanding with BERT](https://cameronrwolfe.substack.com/p/language-understanding-with-bert) — Cameron R. Wolfe's deep dive into BERT
- [Self-Attention from Scratch](https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html) — Sebastian Raschka's step-by-step self-attention tutorial
- [Transformer Encoder-Decoder Architecture (ResearchGate)](https://www.researchgate.net/figure/Transformer-Encoder-Decoder-architecture-taken-from-Vaswani-et-al-9-for-illustration_fig2_338223294) — Architecture diagram from Vaswani et al.
- [Standard Transformer Block (ResearchGate)](https://www.researchgate.net/figure/The-standard-transformer-block_fig2_355391815) — Transformer block diagram

### Andrej Karpathy's Materials (Highly Recommended!)

**Strongly suggested to go through these in detail:**
- [minGPT training code](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) — Essential minimal GPT training snippet
- [Let's build GPT from scratch](https://www.youtube.com/watch?v=XfpMkf4rD6E) — Full video walkthrough of building GPT
- [Andrej Karpathy's YouTube Channel](https://www.youtube.com/@AndrejKarpathy) — Excellent in-depth lectures on neural networks and transformers

## 📝 License

Educational materials for GAIA seminar participants.
