# PDKGC

Code and Data for the paper: "Prompting Disentagled Embeddings for Knowledge Graph Completion with Pre-trained Language Model".

>In this work, we propose a novel method named PDKGC which uses a novel *Prompt* strategy for *Knowledge Graph Completion* on *Pretrained Language Model*, where the PLM is frozen with only the parameters of prompts tunable. \
Our proposed PDKGC builds prompts using a **hard task prompt** and a **disentangled structure prompt**. The hard task prompt is a pre-defined template containing the $\texttt{[MASK]}$ token to reformulate the KGC task as a token prediction task. The disentangled structure prompt is a series of trainable token vectors generated from disentangled entity embeddings and relation embeddings. \
Extensive evaluation on multiple benchmarks has shown the effectiveness of PDKGC's techniques and its better performance compared with the state-of-the-art methods that support frozen PLM for KGC as well as traditional KGC methods.

### Requirements

The model is developed using PyTorch with environment requirements provided in `requirements.txt`.

### Dataset Illustrations

- We use FB15k-237, WN18RR datasets for knowledge graph link prediction.
- FB15k-237 and WN18RR are included in the `data` directory.

### Model Illustrations

We provide the codes for our PDKGC and a baseline of MEM_KGC. We also provide the codes for two ablation studies, without Disentangled Embeddings and without Text Prediction, in the folder `PDKGC_without_Disen` and `PDKGC_without_TP`, respectively.

### Training and Testing

To view specific training commands, please go to the README.md file in the corresponding subdirectory.
