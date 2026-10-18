# Continuum Memory Systems: Multi-Frequency Memory Blocks for Reduced Forgetting in Transformers

Most language models are static after training: when you train them on new tasks, they tend to overwrite what they learned before (*catastrophic forgetting*). Google's 2025 *Nested Learning* paper proposed **Continuum Memory Systems (CMS)**, which replace a transformer's feedforward layers with a chain of memory blocks that update at different frequencies: fast blocks for short-term learning, slow blocks for long-term retention.

In this project I isolated CMS from the paper's other contributions and tested it on its own. I replaced the feedforward layers in all six of DistilBERT's transformer blocks with CMS blocks (three sub-blocks, initialized from DistilBERT's pretrained weights, with update intervals of 1, 2, and 4 steps). I then trained the model sequentially across the 10 domains of the CLINC150 intent-classification dataset, comparing against standard DistilBERT across seven experimental configurations.

**Key findings**
- CMS was seen to reduce forgetting, measured by Backward Transfer (BWT). 
- Forgetting was reduced, not eliminated. Both models still lost most of their knowledge of earlier domains.
- Retention came at a cost: every CMS configuration learned each new domain less well than the base model (lower diagonal accuracy), and CMS needed more epochs to converge.

**Takeaway:** CMS is a promising direction, but replacing feedforward layers with CMS alone wasn't enough to meaningfully solve catastrophic forgetting, and it trades away short-term learning. The *Nested Learning* paper pairs CMS with more expressive optimizers and a self-modifying architecture, so testing those together is the natural next step.

[Full paper](continuum_memory_systems_paper.pdf) · [Notebook](continuum_memory_systems_notebook.ipynb)

*Self-directed final project for EN.705.744 at Johns Hopkins University (Spring 2026).* 

## Running the Notebook

- Developed on **Google Colab** with an A100 GPU.
- No local data is needed: the **CLINC150** dataset (`clinc_oos`, "plus") and **DistilBERT** (`distilbert/distilbert-base-uncased`) download automatically from Hugging Face.
- The Google Drive mount cell near the top is Colab-specific and isn't used by the rest of the notebook. Skip it if running elsewhere.
- Main libraries: `torch`, `transformers`, `datasets`, `pandas`, `matplotlib`.

## Reference

A. Behrouz, M. Razaviyayn, P. Zhong, and V. Mirrokni, "Nested Learning: The Illusion of Deep Learning Architectures," arXiv:2512.24695, 2025.
