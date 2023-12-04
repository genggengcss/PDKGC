<!-- [Results Link Google Doc](https://docs.google.com/spreadsheets/d/16UMyMhjM57x00n8SjMExdmrZxwpCv54pCX9Zb8ofUt0/edit?usp=sharing) -->

### Running Commands
#### for WN18RR

**1. for training**

```bash
# for BERT-large, ConvE
python run.py -epoch 200 -name BERT_large_ConvE_WN18RR -mi_drop -lr 1e-4 -batch 64 -test_batch 64 -num_workers 4 -k_w 10 -k_h 20 -embed_dim 200 -data WN18RR -num_factors 2 -gpu $GPU_number
# for RoBERTa-base, TransE
python run.py -epoch 200 -name RoBERTa_base_TransE_WN18RR -mi_drop -lr 1e-4 -batch 64 -test_batch 64 -num_workers 4 -embed_dim 200 -data WN18RR -pretrained_model roberta_base -score_func TransE -num_factors 2 -gpu $GPU_number
```

**2. for continue training, specify load_epoch, load_path and load type**

```bash
# for BERT-large, ConvE
python run.py -epoch 200 -name BERT_large_ConvE_WN18RR -mi_drop -lr 1e-4 -batch 64 -test_batch 64 -num_workers 4 -k_w 10 -k_h 20 -embed_dim 200 -data WN18RR -num_factors 2 -gpu $GPU_number -load_epoch $epoch -load_type $checkpoint_type -load_path $checkpoint_name
# for RoBERTa-base, TransE
python run.py -epoch 200 -name RoBERTa_base_TransE_WN18RR -mi_drop -lr 1e-4 -batch 64 -test_batch 64 -num_workers 4 -embed_dim 200 -data WN18RR -pretrained_model roberta_base -score_func TransE -num_factors 2 -gpu $GPU_number -load_epoch $epoch -load_type $checkpoint_type -load_path $checkpoint_name
```

`$checkpoint_name` is consisted the pa

**3. for testing**
the same as continue training but add `test` parameter


#### for FB15K-237
**1. for training**
```
python run.py -epoch 2000 -name ConvE_FB15k_K4_D200_club_b_mi_drop_200d_llm -mi_drop -lr 1e-4 -batch 64 -test_batch 64 -num_workers 4 -k_w 10 -k_h 20 -embed_dim 200 -gpu 5
```