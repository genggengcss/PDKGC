## Fine-tune & Frozen LM, BERT-base & BERT-large

**fine-tune with default learning rates, i.e., bert_lr: 1e-5, model_lr: 5e-4 **

**frozen with learning rate, i.e., model_lr: 5e-4 (can tune)**

#### Running commands
**1. for training**
```
python run.py -epoch 200 -name LM_WN18RR_fine_tune -batch 16 -test_batch 16 -num_workers 4 -data WN18RR -gpu 5 -pretrained_model bert_large -fine_tune
python run.py -epoch 200 -name LM_WN18RR_frozen -batch 16 -test_batch 16 -num_workers 4 -data WN18RR -gpu 5 -pretrained_model bert_large
```

**2. for continue training, specify load_epoch and load_path**
```
python run.py -epoch 200 -name LM_WN18RR_fine_tune -batch 16 -test_batch 16 -num_workers 4 -data WN18RR -gpu 5 -pretrained_model bert_large -fine_tune -load_epoch 48 -load_path LM_WN18RR_12_10_2023_15:55:43
```

**3. for testing, set params: test and load_path**
```
python run.py -epoch 200 -name LM_WN18RR_fine_tune -batch 16 -test_batch 16 -num_workers 4 -data WN18RR -gpu 5 -pretrained_model bert_large -fine_tune -test -load_path LM_WN18RR_12_10_2023_15:55:43
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


