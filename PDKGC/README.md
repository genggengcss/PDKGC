<!-- [Results Link Google Doc](https://docs.google.com/spreadsheets/d/16UMyMhjM57x00n8SjMExdmrZxwpCv54pCX9Zb8ofUt0/edit?usp=sharing) -->

### Running Commands
<!-- #### for WN18RR -->

To train the model, please specify the PLMs and score functions used for training at first. You can choose any combination of PLM and score function from the table below.

<table>
<tr>
    <td><strong>PLMs</strong></td>
    <td>bert_base</td>
    <td>bert_large</td>
    <td>roberta_base</td>
    <td>roberta_large</td>
</tr>
<tr>
    <td><strong>score functions</strong></td>
    <td>ConvE</td>
    <td>TranE</td>
    <td>DistMult</td>
    <td></td>
</tr>
</table>

Taking BERT-large, ConvE and RoBERTa-base, TranE as examples:

```bash
# for WN18RR
## for BERT-large, ConvE
python run.py -epoch 200 -name BERT_large_ConvE_WN18RR -mi_drop -lr 1e-4 -batch 64 -test_batch 64 -num_workers 4 -k_w 10 -k_h 20 -embed_dim 200 -data WN18RR -num_factors 2 -gpu $GPU_number -loss_weight
## for RoBERTa-base, TransE
python run.py -epoch 200 -name RoBERTa_base_TransE_WN18RR -mi_drop -lr 1e-4 -batch 64 -test_batch 64 -num_workers 4 -embed_dim 200 -data WN18RR -pretrained_model roberta_base -score_func TransE -num_factors 2 -gpu $GPU_number -loss_weight

# for FB15k-237
## for BERT-large, ConvE
python run.py -epoch 80 -name BERT_large_ConvE_FB15k237 -mi_drop -lr 1e-4 -batch 64 -test_batch 64 -num_workers 4 -k_w 10 -k_h 20 -embed_dim 200 -gpu $GPU_number -loss_weight
## for RoBERTa-base, TransE
python run.py -epoch 80 -name RoBERTa_base_TransE_FB15k237 -mi_drop -lr 1e-4 -batch 64 -test_batch 64 -num_workers 4 -embed_dim 200 -gpu $GPU_number -loss_weight
```

To continue training, specify load_epoch, load_path and load type:

```bash
# for WN18RR
## for BERT-large, ConvE
python run.py -epoch 200 -name BERT_large_ConvE_WN18RR -mi_drop -lr 1e-4 -batch 64 -test_batch 64 -num_workers 4 -k_w 10 -k_h 20 -embed_dim 200 -data WN18RR -num_factors 2 -gpu $GPU_number -loss_weight -load_epoch $epoch -load_type $checkpoint_type -load_path $checkpoint_name
## for RoBERTa-base, TransE
python run.py -epoch 200 -name RoBERTa_base_TransE_WN18RR -mi_drop -lr 1e-4 -batch 64 -test_batch 64 -num_workers 4 -embed_dim 200 -data WN18RR -pretrained_model roberta_base -score_func TransE -num_factors 2 -gpu $GPU_number -loss_weight -load_epoch $epoch -load_type $checkpoint_type -load_path $checkpoint_name

# for FB15k-237
## for BERT-large, ConvE
python run.py -epoch 80 -name BERT_large_ConvE_FB15k237 -mi_drop -lr 1e-4 -batch 64 -test_batch 64 -num_workers 4 -k_w 10 -k_h 20 -embed_dim 200 -gpu $GPU_number -loss_weight -load_epoch $epoch -load_type $checkpoint_type -load_path $checkpoint_name
## for RoBERTa-base, TransE
python run.py -epoch 80 -name RoBERTa_base_TransE_FB15k237 -mi_drop -lr 1e-4 -batch 64 -test_batch 64 -num_workers 4 -embed_dim 200 -gpu $GPU_number -loss_weight -load_epoch $epoch -load_type $checkpoint_type -load_path $checkpoint_name
```

To test the model, the same as continue training but add `test` parameter

```bash
# for WN18RR
## for BERT-large, ConvE
python run.py -epoch 200 -name BERT_large_ConvE_WN18RR -mi_drop -lr 1e-4 -batch 64 -test_batch 64 -num_workers 4 -k_w 10 -k_h 20 -embed_dim 200 -data WN18RR -num_factors 2 -gpu $GPU_number -loss_weight -load_epoch $epoch -load_type $checkpoint_type -load_path $checkpoint_name -test
## for RoBERTa-base, TransE
python run.py -epoch 200 -name RoBERTa_base_TransE_WN18RR -mi_drop -lr 1e-4 -batch 64 -test_batch 64 -num_workers 4 -embed_dim 200 -data WN18RR -pretrained_model roberta_base -score_func TransE -num_factors 2 -gpu $GPU_number -loss_weight -load_epoch $epoch -load_type $checkpoint_type -load_path $checkpoint_name -test

# for FB15k-237
## for BERT-large, ConvE
python run.py -epoch 80 -name BERT_large_ConvE_FB15k237 -mi_drop -lr 1e-4 -batch 64 -test_batch 64 -num_workers 4 -k_w 10 -k_h 20 -embed_dim 200 -gpu $GPU_number -loss_weight -load_epoch $epoch -load_type $checkpoint_type -load_path $checkpoint_name -test
## for RoBERTa-base, TransE
python run.py -epoch 80 -name RoBERTa_base_TransE_FB15k237 -mi_drop -lr 1e-4 -batch 64 -test_batch 64 -num_workers 4 -embed_dim 200 -gpu $GPU_number -loss_weight -load_epoch $epoch -load_type $checkpoint_type -load_path $checkpoint_name -test
```

<!-- #### for FB15K-237
**1. for training** -->
<!-- ```
python run.py -epoch 2000 -name ConvE_FB15k_K4_D200_club_b_mi_drop_200d_llm -mi_drop -lr 1e-4 -batch 64 -test_batch 64 -num_workers 4 -k_w 10 -k_h 20 -embed_dim 200 -gpu 5
``` -->


- `$GPU_number` is the order number of one avalible GPU.
- `$epoch` is the specific epoch where training stopped.
- `$checkpoint_type` is the model type ranged in 'combine', 'struc' and 'text'.
- `$checkpoint_name` is the model name you give at parameter `name`, followed by the start time of first training.
