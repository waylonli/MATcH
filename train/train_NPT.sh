ANNO=conservation
INPUT=both
GLOB=True
MODELPATH=NPT_${ANNO}_${INPUT}_local
TRAIN_SET=./datasets/${ANNO}_train
DEV_SET=./datasets/${ANNO}_dev
TEST_SET=./datasets/${ANNO}_test

python neural_model.py train ${MODELPATH} \
/bask/projects/j/jlxi8926-auto-sum/waylon/mathbert/tasks/theory-proof-matching/anonymized_dataset/${ANNO}_anno_train \
/bask/projects/j/jlxi8926-auto-sum/waylon/mathbert/tasks/theory-proof-matching/anonymized_dataset/${ANNO}_anno_dev \
--seed 10000 -v 40 -i 400 -N 60 -L softmax -W 300 -d 2 --dk 128 --n-heads 4 --max-length 200 --optimizer asgd -l 0.005 --gpu 0 --input ${INPUT} --glob ${GLOB}

python neural_model.py eval ${MODELPATH} \
/bask/projects/j/jlxi8926-auto-sum/waylon/mathbert/tasks/theory-proof-matching/anonymized_dataset/zero_anno_test \
./maximin_zero_anno_both_local/zero_anno_eval_output_full --gpu 0 --input ${INPUT} --glob ${GLOB}
