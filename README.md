# MATcH

Supporting code for the EACL 2023 paper [BERT is not The Count: Learning to Match Mathematical Statements with Proofs](https://aclanthology.org/2023.eacl-main.260/)

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

Once finish downloading the dataset, unzip the three directories "mixed", "pb" and "unmixed" to the `datasets` folder. 

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
bash convert_to_csv.sh
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

|Hyperparameters|Description|
|---|---|
|`MODELPATH`|Path to save the model|
|`PRETRAINPATH`|Path to the pretrained language model|
|`SYM`|Symbol replacement method|
|`INPUT`|Input type, can be [both, math, text]|
|`--seed`|Random seed|
|`-v`|Logger verbosity, higher is quieter|
|`-i`|Training epochs|
|`-N`|Batch size|
|`-L`|Loss for local cross entropy. sigmoid: binary cross entropy|
|`-W`|Encoder dimension|
|`-d`|Encoder depth|
|`--dk`|Encoder query dimension|
|`--n-heads`|Number of attention heads|
|`--max-length`|Max length for self attention sequence (to avoid out of memory errors)|
|`--input`|Input type, can be [both, math, text]|
|`--optimizer`|Optimizer|
|`-l`|Learning rate|
|`--gpu`|GPU id|
|`--pretrainpath`|Path to pretrained language model|


```bash
cd train
conda activate bert

SYM=conservation # symbol replacement method
INPUT=both # input type, can be [both, math, text] 

MODELPATH=./scratchbert_${ANNO}_${INPUT}_local
PRETRAINPATH= # path / link for ScratchBERT pretrained language model

python neural_model_bert.py train ${MODELPATH} ../datasets/${SYM}_train.csv ../datasets/${SYM}_dev.csv ../datasets/conservation_dev.csv \
--seed 10000 -v 40 -i 60 -N 16 -L softmax -W 300 -d 2 --dk 128 --n-heads 4 --max-length 200 --input ${INPUT} --optimizer asgd \
-l 2e-3 --gpu 0 --pretrainpath $PRETRAINPATH
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

python neural_model_bert.py train ${MODELPATH} ../datasets/${SYM}_train.csv ../datasets/${SYM}_dev.csv ../datasets/full_dev.csv \
--seed 10000 -v 40 -i 60 -N 16 -L softmax -W 300 -d 2 --dk 128 --n-heads 4 --max-length 200 --input ${INPUT} --optimizer asgd \ 
-l 2e-3 --gpu 0 --pretrainpath $PRETRAINPATH
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
../datasets/${ANNO}_train \
../datasets/${ANNO}_dev \
--seed 10000 -v 40 -i 400 -N 60 -L softmax -W 300 -d 2 --dk 128 --n-heads 4 --max-length 200 --optimizer asgd -l 0.005 --gpu 0 --input ${INPUT}
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
@inproceedings{li-etal-2023-bert,
    title = {BERT Is Not The Count: Learning to Match Mathematical Statements with Proofs},
    author = {Li, Weixian Waylon and Ziser, Yftah and Coavoux, Maximin and Cohen, Shay B.},
    booktitle = {Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics},
    year = {2023},
    address = {Dubrovnik, Croatia},
    publisher = {Association for Computational Linguistics},
    url = {https://aclanthology.org/2023.eacl-main.260/},
    doi = {10.18653/v1/2023.eacl-main.260},
    pages = {3581--3593}
}
```



