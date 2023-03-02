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