# GAIA Seminar: Introduction to Transformers

Interactive seminar on building a GPT-style transformer from scratch using PyTorch.

## 📚 Contents

- **`seminar_materials/`** - Main teaching materials
  - `gpt_walkthrough.py` - Step-by-step walkthrough (run like a notebook)
  - `gpt_model.py` - Clean implementation matching notebook structure
  - `utils_viz.py` - Visualization utilities
  - `live_coding_template.ipynb` - Template for live coding session
  - `seminar_notebook.ipynb` - Interactive notebook with exercises
  - `chkpts/seminar_notebook_v1_completed.ipynb` - Completed reference

- **`files/`** - Supporting documents
  - `conversation_email` - Seminar context and requirements
  - `speaker_guide.md` - Detailed timing and talking points
  - `good_resources.md` - Additional learning resources

## 🚀 Quick Start (Google Colab)

```python
# 1. Clone the repository
!git clone https://github.com/lukabarbakadze/GAIA-Seminar.git
%cd GAIA-Seminar/seminar_materials

# 2. Install dependencies
!pip install torch matplotlib seaborn scikit-learn

# 3. Run the walkthrough
!python gpt_walkthrough.py
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

## 📝 License

Educational materials for GAIA seminar participants.
