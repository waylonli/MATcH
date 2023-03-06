SYM=conservation # symbol replacement method
INPUT=both # input type, can be [both, math, text] 

MODELPATH=./scratchbert_${ANNO}_${INPUT}_local
PRETRAINPATH=/disk/ocean/waylon/MATcH/model_files # path / link for ScratchBERT pretrained language model

python neural_model_bert.py train ${MODELPATH} ../datasets/${SYM}_train.csv ../datasets/${SYM}_dev.csv ../datasets/full_dev.csv \
--seed 10000 -v 40 -i 60 -N 16 -L softmax -W 300 -d 2 --dk 128 --n-heads 4 --max-length 200 --input ${INPUT} --optimizer asgd \
-l 2e-3 --gpu 0 --pretrainpath $PRETRAINPATH