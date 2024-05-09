# PDKGC

Code and Data for the submission: "Prompting Disentagled Embeddings for Knowledge Graph Completion with Pre-trained Language Model".

We Provide two versions of Code for Our model. One is traditional Pytorch version provided in this repository. The Other is Pytorch lightning version provided in repository PDKGC_Pytorch_lightning([link](https://github.com/zeng-orc/PDKGC_Pytorch_lightning))

>In this work, we propose a novel prompt tuning method named PDKGC for KGC problem, which includes a hard task prompt and a disentangled structure prompt for better performing KGC on frozen Pre-trained Language Models (PLMs).
<br>The hard task prompt is a pre-defined template containing the [MASK] token, which reformulates the KGC task as a token prediction task in line with the pre-training tasks of many PLMs.
The disentangled structure prompt is a series of trainable vectors (i.e., soft prompts) generated from disentangled entity embeddings which are learned by a graph learner with selective aggregations over different neighboring entities. It is prepended to the hard task prompt to form the inputs of a frozen PLM.
After encoding by the PLM, for a KG triple to complete, PDKGC not only includes a textual predictor to output a probability distribution over all entities based on the encoded representations of the [MASK] token which has fused relevant structural knowledge, but also includes a structural predictor to simultaneously produce the entity ranks by forwarding the encoded representations of structural soft prompts to a KGE model.
Naturally, their outputs can be further combined for more comprehensive prediction.
<br>Evaluations on two popular KG datasets demonstrate the superiority of our proposed model to another prompt-tuning-based baseline [CSProm-KG](https://arxiv.org/abs/2307.01709), as well as fine-tuned PLM-based methods and traditional structure-based methods.


### Requirements

The model is developed using PyTorch with environment requirements provided in `requirements.txt`.

### Dataset Preparations

- We experiment with FB15k-237, WN18RR datasets for Entity Prediction.
- FB15k-237 and WN18RR are included in the `data` directory.

### Model Illustrations

We provide the codes for our PDKGC and a baseline of [MEM-KGC](https://ieeexplore.ieee.org/document/9540703).

We also publish the codes of two model variants for ablation studies, i.e., `PDKGC_without_Disen` with non-Disentangled Embeddings and `PDKGC_without_TP` with the Text Predictor (TP) removed.

### Training and Testing

To view specific training and testing commands, please go to the README.md file in the corresponding subdirectory.
