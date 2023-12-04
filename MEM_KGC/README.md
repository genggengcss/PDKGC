## Fine-tune & Frozen LM, BERT-base & BERT-large

**fine-tune with default learning rates, i.e., bert_lr: 1e-5, model_lr: 5e-4**

**frozen with learning rate, i.e., model_lr: 5e-4 (can tune)**

#### Running commands

To train the model, please specify which PLMs used for the training. Taking BERT-base as an example:

```bash
# for WN18RR
## fine_tune
python run.py -epoch 200 -name BERT_base_WN18RR_fine_tune -batch 16 -test_batch 16 -num_workers 4 -data WN18RR -gpu $GPU_number -pretrained_model bert_base -fine_tune
## frozen
python run.py -epoch 200 -name BERT_base_WN18RR_frozen -batch 16 -test_batch 16 -num_workers 4 -data WN18RR -gpu $GPU_number -pretrained_model bert_base

# for FB15k-237
## fine_tune
python run.py -epoch 80 -name BERT_base_FB15k237_fine_tune -batch 16 -test_batch 16 -num_workers 4 -gpu $GPU_number -pretrained_model bert_base -fine_tune
## frozen
python run.py -epoch 80 -name BERT_base_FB15k237_frozen -batch 16 -test_batch 16 -num_workers 4 -gpu $GPU_number -pretrained_model bert_base
```

To continue training, specify load_epoch and load_path:

```bash
# for WN18RR
## fine_tune
python run.py -epoch 200 -name BERT_base_WN18RR_fine_tune -batch 16 -test_batch 16 -num_workers 4 -data WN18RR -gpu $GPU_number -pretrained_model bert_base -fine_tune -load_epoch $epoch -load_path $checkpoint_name
## frozen
python run.py -epoch 200 -name BERT_base_WN18RR_frozen -batch 16 -test_batch 16 -num_workers 4 -data WN18RR -gpu $GPU_number -pretrained_model bert_base -load_epoch $epoch -load_path $checkpoint_name

# for FB15k-237
## fine_tune
python run.py -epoch 80 -name BERT_base_FB15k237_fine_tune -batch 16 -test_batch 16 -num_workers 4 -gpu $GPU_number -pretrained_model bert_base -fine_tune -load_epoch $epoch -load_path $checkpoint_name
## frozen
python run.py -epoch 80 -name BERT_base_FB15k237_frozen -batch 16 -test_batch 16 -num_workers 4 -gpu $GPU_number -pretrained_model bert_base -load_epoch $epoch -load_path $checkpoint_name
```

To test the model, set params test and load_path:

```bash
# for WN18RR
## fine_tune
python run.py -epoch 200 -name BERT_base_WN18RR_fine_tune -batch 16 -test_batch 16 -num_workers 4 -data WN18RR -gpu $GPU_number -pretrained_model bert_base -fine_tune -test -load_path $checkpoint_name
## frozen
python run.py -epoch 200 -name BERT_base_WN18RR_frozen -batch 16 -test_batch 16 -num_workers 4 -data WN18RR -gpu $GPU_number -pretrained_model bert_base -test -load_path $checkpoint_name

# for FB15k-237
## fine_tune
python run.py -epoch 80 -name BERT_base_FB15k237_fine_tune -batch 16 -test_batch 16 -num_workers 4 -gpu $GPU_number -pretrained_model bert_base -fine_tune -test -load_path $checkpoint_name
## frozen
python run.py -epoch 80 -name BERT_base_FB15k237_frozen -batch 16 -test_batch 16 -num_workers 4 -gpu $GPU_number -pretrained_model bert_base -test -load_path $checkpoint_name
```

#### Partial Results on WN18RR

| Methods | MRR | MR | Hits@1 | Hits@3 | Hits@10 |
|--------------------------------------|----|-----|--------|--------|---------|
| fine-tune BERT base (best val)       | 0.53409 |  | 0.49094 | 0.55686 | 0.61124 |
| fine-tune BERT base (test)           | 0.53439 |2087.9| 0.48851 | 0.55855 | 0.61726 |
|----------------------------------------|----|-----|--------|--------|---------|
| fine-tune BERT large (best val)      | 0.53316 |  | 0.48748| 0.55669 | 0.61256 |
| fine-tune BERT large (test)           | 0.52994 |  | 0.48261 | 0.55871 | 0.61088 |
|----------------------------------------|----|-----|--------|--------|---------|
|                  | |  |  |  |  |
