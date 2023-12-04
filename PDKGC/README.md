####

[Results Link Google Doc](https://docs.google.com/spreadsheets/d/16UMyMhjM57x00n8SjMExdmrZxwpCv54pCX9Zb8ofUt0/edit?usp=sharing)

### Running Commands
#### for WN18RR

**1. for training**
```
python run.py -epoch 2000 -name ConvE_WN18RR_K2_D200_club_b_mi_drop_200d_llm -mi_drop -lr 1e-4 -batch 64 -test_batch 64 -num_workers 4 -k_w 10 -k_h 20 -embed_dim 200 -data WN18RR -num_factors 2 -gpu 5
```

**2. for continue training, specify load_epoch, load_path and load type**
```
python run.py -epoch 2000 -name ConvE_WN18RR_K2_D200_club_b_mi_drop_200d_llm -mi_drop -lr 1e-4 -batch 64 -test_batch 64 -num_workers 4 -k_w 10 -k_h 20 -embed_dim 200 -data WN18RR -num_factors 2 -gpu 5 -load_epoch 133 -load_type text -load_path ConvE_WN18RR_K2_D200_club_b_mi_drop_200d_llm_14_10_2023_13:53:04
```
**3. for testing**
the same as continue training but add `test` parameter


#### for FB15K-237
**1. for training**
```
python run.py -epoch 2000 -name ConvE_FB15k_K4_D200_club_b_mi_drop_200d_llm -mi_drop -lr 1e-4 -batch 64 -test_batch 64 -num_workers 4 -k_w 10 -k_h 20 -embed_dim 200 -gpu 5
```