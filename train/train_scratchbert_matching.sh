ANNO=partial
INPUT=math

MODELPATH=./${ANNO}_${INPUT}_local
PRETRAINPATH=../pretrain/model_files

# train local model
python neural_model_bert.py train $MODELPATH \
./datasets/${ANNO}_anno_train.csv \
./datasets/${ANNO}_anno_dev.csv \
./datasets/full_anno_dev.csv \
--seed 10000 -v 40 -i 60 -N 16 -L softmax -W 300 -d 2 --dk 128 --n-heads 4 --max-length 200 \
--input ${INPUT} --optimizer asgd -l 2e-3 --gpu 0 \
--pretrainpath ${PRETRAINPATH}

# train global model
#python neural_model_bert.py train $MODELPATH \
#./datasets/${ANNO}_anno_train.csv \
#./datasets/${ANNO}_anno_dev.csv \
#./datasets/full_anno_dev.csv \
#--seed 10000 -v 40 -i 60 -N 16 -G 16 -L softmax -W 300 -d 2 --dk 128 --n-heads 4 --max-length 200 \
#--input ${INPUT} --optimizer asgd -l 2e-3 --gpu 0 \
#--pretrainpath ${PRETRAINPATH}

# eval
python neural_model_bert.py eval $MODELPATH \
./datasets/conservation_test.csv \
. --gpu 0 --max_length 200 --input ${INPUT} \
--pretrainpath ${PRETRAINPATH}