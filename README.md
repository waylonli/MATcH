# MATcH

Supporting code for the paper [BERT is not The Count: Learning to Match Mathematical Statements with Proofs](https://homepages.inf.ed.ac.uk/scohen/eacl23match.pdf)

## Getting start

Clone this repository and install the dependencies:

```bash
git clone https://github.com/waylonli/MATcH.git
cd MATcH
conda env create -f MATcH_env.yml
conda activate bert
```

## Download datasets

https://bollin.inf.ed.ac.uk/match.html

## Training: NPT

If you want to use the same hyperparameter setting in the paper, you can simply run:

```bash
cd train
bash train_NPT.sh
```

If you want to specify any hyperparameters, here's an example for training a local model:

```bash
cd train
conda activate bert

MODELPATH=./trained_NPT_model
ANNO=conservation # can be changed to partial / trans / full

python neural_model.py train ${MODELPATH} \
./datasets/${ANNO}_train \ # training set
./datasets/${ANNO}_dev \ # validation set
--seed 10000 \ # random seed
-v 40 \ # logger verbosity, higher is quieter
-i 400 \ # training epochs
-N 60 \ # batch size
-L softmax \ # loss for local cross entropy. sigmoid: binary cross entropy
-W 300 \ # encoder dimension
-d 2 \ # encoder depth
--dk 128 \ # encoder query dimension
--n-heads 4 \ # number of attention heads
--max-length 200 \ # max length for self attention sequence (to avoid out of memory errors)
--optimizer asgd \ # optimizer
-l 0.005 \ # learning rate
--gpu 0 \ # gpu id
--input ${INPUT} # input type, can be [both, math, text]
```

## Training: MathBERT

If you want to use the same hyperparameter setting in the paper, you can simply run:

```bash
cd train
bash train_mathbert_matching.sh
```

If you want to specify any hyperparameters, here's an example for training a local MathBERT matching model:

```bash
cd train
conda activate bert

ANNO=conservation
INPUT=both

MODELPATH=./mathbert_${ANNO}_${INPUT}_local
PRETRAINPATH=tbs17/MathBERT-custom # path / link for pretrained language model

python neural_model_bert.py train ${MODELPATH} \
./datasets/${ANNO}_train.csv \
./datasets/${ANNO}_dev.csv \
./datasets/full_dev.csv \
--seed 10000 \
-v 40 \
-i 60 \
-N 16 \
-L softmax \
-W 300 \
-d 2 \
--dk 128 \
--n-heads 4 \
--max-length 200 \
--input ${INPUT} \
--optimizer asgd \
-l 2e-3 \
--gpu 0 \
--pretrainpath $PRETRAINPATH
```

## Training: ScratchBERT

Pretrain ScratchBERT on MATcH:

```bash
cd pretrain
bash run_pretrain.sh
```

Train ScratchBERT matching model:

```bash
cd train
bash train_scratchbert_matching.sh
```

If you want to specify the hyperparameter, follow the instruction for training MathBERT matching model, but change the pretrain path to where ScratchBERT model is located.

## Evaluation

Evaluate NPT model:

```bash
cd train

MODELPATH= # the location of the trained matching model
GLOB= # True / False
INPUT= # both / math / text
TESTSET= # test set

python neural_model.py eval ${MODELPATH} \
${TESTSET} \
./eval_output \
--gpu 0 \
--input ${INPUT} \
--glob ${GLOB}
```

## Citation

```latex
@inproceedings{li-23,
    author = "W. Li and Y. Ziser and M. Coavoux and S. B. Cohen",
    title = "BERT is not The Count: Learning to Match Mathematical Statements with Proofs",
    journal = "Proceedings of {EACL}",
    year = "2023"
}
```



