ANNO=conservation
INPUT=both

MODELPATH=/bask/projects/j/jlxi8926-auto-sum/waylon/mathbert/tasks/SimCSE/new_mathbert_${ANNO}_${INPUT}_local
PRETRAINPATH=/bask/projects/j/jlxi8926-auto-sum/waylon/mathbert/tasks/pretrain/baseline/MathBERT-custom

# train local model
python neural_model_bert.py train ${MODELPATH} \
./datasets/${ANNO}_train.csv \
./datasets/${ANNO}_dev.csv \
./datasets/full_dev.csv \
--seed 10000 -v 40 -i 60 -N 16 -L softmax -W 300 -d 2 --dk 128 --n-heads 4 --max-length 200 \
--input ${INPUT} --optimizer asgd -l 2e-3 --gpu 0 \
--pretrainpath $PRETRAINPATH

# train global model
#python neural_model_bert.py train $MODELPATH \
#./datasets/${ANNO}_train.csv \
#./datasets/${ANNO}_dev.csv \
#./datasets/full_dev.csv \
#--seed 10000 -v 40 -i 60 -G 16 -N 16 -L softmax -W 300 -d 2 --dk 128 --n-heads 4 --max-length 200 \
#--input ${INPUT} --optimizer asgd -l 2e-3 --gpu 0 \
#--pretrainpath $PRETRAINPATH

# eval local decoding
python neural_model_bert.py eval $MODELPATH \
./datasets/conservation_test.csv \
./ranking_eval_output --gpu 0 --max_length 200 --input ${INPUT} \
--pretrainpath $PRETRAINPATH

# eval global decoding
#python neural_model_bert.py eval $MODELPATH \
#./datasets/conservation_test.csv \
#./ranking_eval_output --gpu 0 --max_length 200 --input ${INPUT} \
#--pretrainpath $PRETRAINPATH --glob True

