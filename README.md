# MATcH

Supporting code for the EACL 2023 paper [BERT is not The Count: Learning to Match Mathematical Statements with Proofs](https://arxiv.org/abs/2102.02110)

We develop a model to match mathematical statements to their corresponding proofs. This task can be used in areas such as mathematical information retrieval. Our work comes with a dataset for this task, including a large number of statement-proof pairs in different areas of mathematics.

## Getting started

Clone this repository and install the dependencies:

```bash
git clone https://github.com/waylonli/MATcH.git
cd MATcH
conda env create -f MATcH_env.yml
conda activate bert
```

If conda does not manage to automatically install PyTorch and cuda. Please install the pip packages manually by:

```bash
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install transformers==4.2.1
pip install lap
pip install datasets==2.3.2
pip install scikit-learn
pip install lxml
pip install spacy
python -m spacy download en_core_web_sm
```

## Download datasets

Available at https://bollin.inf.ed.ac.uk/match.html (or see dataset/ for README)

Once finish downloading the dataset, unzip the three directories "mixed", "pb" and "unmixed" to the "datasets" folder. 

If you need a csv version of dataset for training ScratchBERT and MathBERT matching model, do the following:
```bash
cd train

SYM=conservation # conservation / partial / full / trans
SPLIT=unmixed # mixed / unmixed
DATASETPATH=../datasets

python ../train/convert_to_csv.py ${DATASETPATH}/${SPLIT}/${SPLIT}_${SYM}_train ${DATASETPATH}/${SYM}_train.csv
python ../train/convert_to_csv.py ${DATASETPATH}/${SPLIT}/${SPLIT}_${SYM}_dev ${DATASETPATH}/${SYM}_dev.csv
python ../train/convert_to_csv.py ${DATASETPATH}/${SPLIT}/${SPLIT}_${SYM}_test ${DATASETPATH}/${SYM}_test.csv
```

You can also edit the `./train/convert_to_csv.sh` script and simply run:
```bash
cd train
convert_to_csv.sh
```

## Training: ScratchBERT

### Pretraining

Pretrain ScratchBERT on MATcH:

```bash
cd pretrain
bash run_pretrain.sh
```

**OR**

You can also download our pretrained version of ScratchBERT if you don't want to re-pretrain the language model: https://bollin.inf.ed.ac.uk/match.html

### After Pretraining

Train ScratchBERT matching model:

```bash
cd train
bash train_scratchbert_matching.sh
```

If you want to specify the hyperparameter, here's an example for training a local ScratchBERT matching model:

```bash
cd train
conda activate bert

SYM=conservation # symbol replacement method
INPUT=both # input type, can be [both, math, text] 

MODELPATH=./scratchbert_${ANNO}_${INPUT}_local
PRETRAINPATH=./model_files # path / link for ScratchBERT pretrained language model

python neural_model_bert.py train ${MODELPATH} \
./datasets/${SYM}_train.csv \
./datasets/${SYM}_dev.csv \
./datasets/full_dev.csv \
--seed 10000 \ # random seed
-v 40 \ # logger verbosity, higher is quieter
-i 60 \ # training epochs
-N 16 \ # batch size
-L softmax \ # loss for local cross entropy. sigmoid: binary cross entropy
-W 300 \ # encoder dimension
-d 2 \ # encoder depth
--dk 128 \ # encoder query dimension
--n-heads 4 \ # number of attention heads
--max-length 200 \ # max length for self attention sequence (to avoid out of memory errors)
--input ${INPUT} \ # input type, can be [both, math, text]
--optimizer asgd \ # optimizer
-l 2e-3 \ # learning rate
--gpu 0 \ # gpu id
--pretrainpath $PRETRAINPATH # path to pretrained language model
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

SYM=conservation # symbol replacement method
INPUT=both # input type, can be [both, math, text] 

MODELPATH=./mathbert_${ANNO}_${INPUT}_local
PRETRAINPATH=tbs17/MathBERT-custom # path / link for pretrained language model

python neural_model_bert.py train ${MODELPATH} \
./datasets/${SYM}_train.csv \
./datasets/${SYM}_dev.csv \
./datasets/full_dev.csv \
--seed 10000 \ # random seed
-v 40 \ # logger verbosity, higher is quieter
-i 60 \ # training epochs
-N 16 \ # batch size
-L softmax \ # loss for local cross entropy. sigmoid: binary cross entropy
-W 300 \ # encoder dimension
-d 2 \ # encoder depth
--dk 128 \ # encoder query dimension
--n-heads 4 \ # number of attention heads
--max-length 200 \ # max length for self attention sequence (to avoid out of memory errors)
--input ${INPUT} \ # input type, can be [both, math, text]
--optimizer asgd \ # optimizer
-l 2e-3 \ # learning rate
--gpu 0 \ # gpu id
--pretrainpath $PRETRAINPATH
```

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
SYM=conservation # can be changed to partial / trans / full

python neural_model.py train ${MODELPATH} \
./datasets/${SYM}_train \ # training set
./datasets/${SYM}_dev \ # validation set
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

```
@inproceedings{li-23,
    author = "W. Li and Y. Ziser and M. Coavoux and S. B. Cohen",
    title = "BERT is not The Count: Learning to Match Mathematical Statements with Proofs",
    journal = "Proceedings of {EACL}",
    year = "2023"
}
```



