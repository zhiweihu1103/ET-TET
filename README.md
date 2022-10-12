# Transformer-based Entity Typing in Knowledge Graphs
#### This repo provides the source code & data of our paper: Transformer-based Entity Typing in Knowledge Graphs (EMNLP2022).

## Dependencies
* conda create -n tet python=3.7 -y
* PyTorch 1.8.1
* transformers 4.7.0
* pytorch-pretrained-bert 0.6.2

## Running the code
### Dataset
* Download the datasets from [Here](https://drive.google.com/drive/folders/120QIGxsGQXfH6Rd8wJe7i57gg8dlx7l2?usp=sharing).
* Create the root directory ./data and put the dataset in.

### Training model
#### For FB15kET dataset
```python
export DATASET=FB15kET
export SAVE_DIR_NAME=FB15kET
export LOG_PATH=./logs/FB15kET.out
export HIDDEN_DIM=100
export TEMPERATURE=0.5
export LEARNING_RATE=0.001
export TRAIN_BATCH_SIZE=128
export MAX_EPOCH=500
export VALID_EPOCH=25
export BETA=1
export LOSS=SFNA

export PAIR_POOLING=avg
export SAMPLE_ET_SIZE=3
export SAMPLE_KG_SIZE=7
export SAMPLE_ENT2PAIR_SIZE=6
export WARM_UP_STEPS=50
export TT_ABLATION=all

CUDA_VISIBLE_DEVICES=0 python ./run.py --dataset $DATASET --save_path $SAVE_DIR_NAME --hidden_dim $HIDDEN_DIM --temperature $TEMPERATURE --lr $LEARNING_RATE \
  --train_batch_size $TRAIN_BATCH_SIZE --cuda --max_epoch $MAX_EPOCH --valid_epoch $VALID_EPOCH --beta $BETA --loss $LOSS \
  --pair_pooling $PAIR_POOLING --sample_et_size $SAMPLE_ET_SIZE --sample_kg_size $SAMPLE_KG_SIZE --sample_ent2pair_size $SAMPLE_ENT2PAIR_SIZE --warm_up_steps $WARM_UP_STEPS \
  --tt_ablation $TT_ABLATION \
  > $LOG_PATH 2>&1 &
```
#### For YAGO43kET dataset
```python
export DATASET=YAGO43kET
export SAVE_DIR_NAME=YAGO43kET
export LOG_PATH=./logs/YAGO43kET.out
export HIDDEN_DIM=100
export TEMPERATURE=0.5
export LEARNING_RATE=0.001
export TRAIN_BATCH_SIZE=128
export MAX_EPOCH=500
export VALID_EPOCH=25
export BETA=1
export LOSS=SFNA

export PAIR_POOLING=avg
export SAMPLE_ET_SIZE=3
export SAMPLE_KG_SIZE=8
export SAMPLE_ENT2PAIR_SIZE=6
export WARM_UP_STEPS=50
export TT_ABLATION=all

CUDA_VISIBLE_DEVICES=1 python ./run.py --dataset $DATASET --save_path $SAVE_DIR_NAME --hidden_dim $HIDDEN_DIM --temperature $TEMPERATURE --lr $LEARNING_RATE \
  --train_batch_size $TRAIN_BATCH_SIZE --cuda --max_epoch $MAX_EPOCH --valid_epoch $VALID_EPOCH --beta $BETA --loss $LOSS \
  --pair_pooling $PAIR_POOLING --sample_et_size $SAMPLE_ET_SIZE --sample_kg_size $SAMPLE_KG_SIZE --sample_ent2pair_size $SAMPLE_ENT2PAIR_SIZE --warm_up_steps $WARM_UP_STEPS \
  --tt_ablation $TT_ABLATION \
  > $LOG_PATH 2>&1 &
```

* **Note:** Before running, you need to create the ./logs folder first.
## Acknowledgement
We refer to the code of [CET](https://github.com/CCIIPLab/CET). Thanks for their contributions.
