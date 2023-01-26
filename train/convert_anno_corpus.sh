#!/bin/bash
#SBATCH --qos epsrc
#SBATCH --account=jlxi8926-auto-sum
#SBATCH --time=4:0:0
#SBATCH --nodes=1

module purge
module load baskerville

#! Insert additional module load commands after this line if needed:
source ~/.bashrc
conda activate bert

DIR="/bask/projects/j/jlxi8926-auto-sum/waylon/mathbert/tasks/SimCSE/datasets/"
MODE="pair"
AUG="naive"
ANNO="partial"
NUM="50k"
echo  $MODE $AUG $ANNO
#python /bask/projects/j/jlxi8926-auto-sum/waylon/mathbert/tasks/theory-proof-matching/maximin_ori/corpus_utils.py /bask/projects/j/jlxi8926-auto-sum/waylon/dataset_09_05 /bask/projects/j/jlxi8926-auto-sum/waylon/zero_anno_no_filter --preprocessing f --no_filter True
#python /bask/projects/j/jlxi8926-auto-sum/waylon/mathbert/tasks/theory-proof-matching/statement_proof_matching-master/convert_to_csv.py /bask/projects/j/jlxi8926-auto-sum/waylon/zero_anno_no_filter /bask/projects/j/jlxi8926-auto-sum/waylon/zero_anno_no_filter.csv

#python /bask/projects/j/jlxi8926-auto-sum/waylon/mathbert/tasks/theory-proof-matching/maximin_ori/corpus_utils.py ${DIR}${MODE}_${AUG}_${ANNO}_positive ${DIR}${MODE}_${AUG}_${ANNO}_positive.txt --preprocessing f --no_filter True
#python /bask/projects/j/jlxi8926-auto-sum/waylon/mathbert/tasks/theory-proof-matching/maximin_ori/corpus_utils.py ${DIR}${MODE}_${AUG}_${ANNO}_negative ${DIR}${MODE}_${AUG}_${ANNO}_negative.txt --preprocessing f --no_filter True
#python /bask/projects/j/jlxi8926-auto-sum/waylon/mathbert/tasks/theory-proof-matching/statement_proof_matching-master/convert_to_csv.py ${DIR}${MODE}_${AUG}_${ANNO}_positive.txt ${DIR}${MODE}_${AUG}_${ANNO}_positive.csv
#python /bask/projects/j/jlxi8926-auto-sum/waylon/mathbert/tasks/theory-proof-matching/statement_proof_matching-master/convert_to_csv.py ${DIR}${MODE}_${AUG}_${ANNO}_negative.txt ${DIR}${MODE}_${AUG}_${ANNO}_negative.csv
#python /bask/projects/j/jlxi8926-auto-sum/waylon/mathbert/preprocess/generate_simcse_dataset.py ${MODE} ${DIR}${MODE}_${AUG}_${ANNO}_aug_train.csv --file_1 ${DIR}${MODE}_${AUG}_${ANNO}_positive.csv --file_2 ${DIR}${MODE}_${AUG}_${ANNO}_negative.csv

#python /bask/projects/j/jlxi8926-auto-sum/waylon/mathbert/tasks/theory-proof-matching/maximin_ori/corpus_utils.py ${DIR}${ANNO}_pb_no_filter /bask/projects/j/jlxi8926-auto-sum/waylon/mathbert/tasks/theory-proof-matching/anonymized_dataset/${ANNO}_pb_no_filter --preprocessing f
#python /bask/projects/j/jlxi8926-auto-sum/waylon/mathbert/tasks/theory-proof-matching/statement_proof_matching-master/convert_to_csv.py /bask/projects/j/jlxi8926-auto-sum/waylon/mathbert/tasks/theory-proof-matching/anonymized_dataset/${ANNO}_pb_no_filter_train /bask/projects/j/jlxi8926-auto-sum/waylon/mathbert/tasks/theory-proof-matching/anonymized_dataset/${ANNO}_pb_no_filter_train.csv
#python /bask/projects/j/jlxi8926-auto-sum/waylon/mathbert/tasks/theory-proof-matching/statement_proof_matching-master/convert_to_csv.py /bask/projects/j/jlxi8926-auto-sum/waylon/mathbert/tasks/theory-proof-matching/anonymized_dataset/${ANNO}_pb_no_filter_dev /bask/projects/j/jlxi8926-auto-sum/waylon/mathbert/tasks/theory-proof-matching/anonymized_dataset/${ANNO}_pb_no_filter_dev.csv
#python /bask/projects/j/jlxi8926-auto-sum/waylon/mathbert/tasks/theory-proof-matching/statement_proof_matching-master/convert_to_csv.py /bask/projects/j/jlxi8926-auto-sum/waylon/mathbert/tasks/theory-proof-matching/anonymized_dataset/${ANNO}_pb_no_filter_test /bask/projects/j/jlxi8926-auto-sum/waylon/mathbert/tasks/theory-proof-matching/anonymized_dataset/${ANNO}_pb_no_filter_test.csv
#python /bask/projects/j/jlxi8926-auto-sum/waylon/mathbert/tasks/theory-proof-matching/statement_proof_matching-master/convert_to_csv.py /bask/projects/j/jlxi8926-auto-sum/waylon/mathbert/tasks/theory-proof-matching/anonymized_dataset/${ANNO}_pb_no_filter_train /bask/projects/j/jlxi8926-auto-sum/waylon/mathbert/tasks/theory-proof-matching/anonymized_dataset/${ANNO}_pb_no_filter_train_math_only.csv --math_only
#python /bask/projects/j/jlxi8926-auto-sum/waylon/mathbert/tasks/theory-proof-matching/statement_proof_matching-master/convert_to_csv.py /bask/projects/j/jlxi8926-auto-sum/waylon/mathbert/tasks/theory-proof-matching/anonymized_dataset/${ANNO}_pb_no_filter_dev /bask/projects/j/jlxi8926-auto-sum/waylon/mathbert/tasks/theory-proof-matching/anonymized_dataset/${ANNO}_pb_no_filter_dev_math_only.csv --math_only
#python /bask/projects/j/jlxi8926-auto-sum/waylon/mathbert/tasks/theory-proof-matching/statement_proof_matching-master/convert_to_csv.py /bask/projects/j/jlxi8926-auto-sum/waylon/mathbert/tasks/theory-proof-matching/anonymized_dataset/${ANNO}_pb_no_filter_test /bask/projects/j/jlxi8926-auto-sum/waylon/mathbert/tasks/theory-proof-matching/anonymized_dataset/${ANNO}_pb_no_filter_test_math_only.csv --math_only
#python /bask/projects/j/jlxi8926-auto-sum/waylon/mathbert/tasks/theory-proof-matching/statement_proof_matching-master/convert_to_csv.py /bask/projects/j/jlxi8926-auto-sum/waylon/mathbert/tasks/theory-proof-matching/anonymized_dataset/${ANNO}_pb_no_filter_train /bask/projects/j/jlxi8926-auto-sum/waylon/mathbert/tasks/theory-proof-matching/anonymized_dataset/${ANNO}_pb_no_filter_train_text_only.csv --text_only
#python /bask/projects/j/jlxi8926-auto-sum/waylon/mathbert/tasks/theory-proof-matching/statement_proof_matching-master/convert_to_csv.py /bask/projects/j/jlxi8926-auto-sum/waylon/mathbert/tasks/theory-proof-matching/anonymized_dataset/${ANNO}_pb_no_filter_dev /bask/projects/j/jlxi8926-auto-sum/waylon/mathbert/tasks/theory-proof-matching/anonymized_dataset/${ANNO}_pb_no_filter_dev_text_only.csv --text_only
#python /bask/projects/j/jlxi8926-auto-sum/waylon/mathbert/tasks/theory-proof-matching/statement_proof_matching-master/convert_to_csv.py /bask/projects/j/jlxi8926-auto-sum/waylon/mathbert/tasks/theory-proof-matching/anonymized_dataset/${ANNO}_pb_no_filter_test /bask/projects/j/jlxi8926-auto-sum/waylon/mathbert/tasks/theory-proof-matching/anonymized_dataset/${ANNO}_pb_no_filter_test_text_only.csv --text_only

python /bask/projects/j/jlxi8926-auto-sum/waylon/mathbert/tasks/theory-proof-matching/maximin_ori/corpus_utils.py ${DIR}${MODE}_${NUM}_${AUG}_${ANNO}_positive ${DIR}${MODE}_${NUM}_${AUG}_${ANNO}_positive.txt --preprocessing f --no_filter True
python /bask/projects/j/jlxi8926-auto-sum/waylon/mathbert/tasks/theory-proof-matching/maximin_ori/corpus_utils.py ${DIR}${MODE}_${NUM}_${AUG}_${ANNO}_negative ${DIR}${MODE}_${NUM}_${AUG}_${ANNO}_negative.txt --preprocessing f --no_filter True
python /bask/projects/j/jlxi8926-auto-sum/waylon/mathbert/tasks/theory-proof-matching/statement_proof_matching-master/convert_to_csv.py ${DIR}${MODE}_${NUM}_${AUG}_${ANNO}_positive.txt ${DIR}${MODE}_${NUM}_${AUG}_${ANNO}_positive.csv
python /bask/projects/j/jlxi8926-auto-sum/waylon/mathbert/tasks/theory-proof-matching/statement_proof_matching-master/convert_to_csv.py ${DIR}${MODE}_${NUM}_${AUG}_${ANNO}_negative.txt ${DIR}${MODE}_${NUM}_${AUG}_${ANNO}_negative.csv
python /bask/projects/j/jlxi8926-auto-sum/waylon/mathbert/preprocess/generate_simcse_dataset.py ${MODE} ${DIR}${MODE}_${NUM}_${AUG}_${ANNO}_aug_train.csv --file_1 ${DIR}${MODE}_${NUM}_${AUG}_${ANNO}_positive.csv --file_2 ${DIR}${MODE}_${NUM}_${AUG}_${ANNO}_negative.csv
