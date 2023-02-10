#!/bin/bash
#SBATCH --qos epsrc
#SBATCH --account=jlxi8926-auto-sum
#SBATCH --time=3-0:0:0
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mem=160G
#SBATCH --output=/bask/projects/j/jlxi8926-auto-sum/waylon/mathbert/tasks/SimCSE/output_partial_math_local.output
module purge
module load baskerville

#! Insert additional module load commands after this line if needed:
module load CUDA/11.1.1-GCC-10.2.0
source ~/.bashrc
conda activate bert

ANNO=partial
INPUT=math

MODELPATH=/bask/projects/j/jlxi8926-auto-sum/waylon/mathbert/tasks/SimCSE/new_${ANNO}_${INPUT}_local
PRETRAINPATH=/bask/projects/j/jlxi8926-auto-sum/waylon/mathbert/tasks/pretrain/train_from_scratch/model_files

python neural_model_bert.py train $MODELPATH \
/bask/projects/j/jlxi8926-auto-sum/waylon/mathbert/tasks/theory-proof-matching/anonymized_dataset/${ANNO}_anno_train.csv \
/bask/projects/j/jlxi8926-auto-sum/waylon/mathbert/tasks/theory-proof-matching/anonymized_dataset/${ANNO}_anno_dev.csv \
/bask/projects/j/jlxi8926-auto-sum/waylon/mathbert/tasks/theory-proof-matching/anonymized_dataset/full_anno_dev.csv \
--seed 10000 -v 40 -i 60 -N 16 -L softmax -W 300 -d 2 --dk 128 --n-heads 4 --max-length 200 \
--input ${INPUT} --optimizer asgd -l 2e-3 --gpu 0 \
--pretrainpath ${PRETRAINPATH}

python neural_model_bert.py eval $MODELPATH \
/bask/projects/j/jlxi8926-auto-sum/waylon/mathbert/tasks/theory-proof-matching/anonymized_dataset/zero_anno_test.csv \
. --gpu 0 --max_length 200 --input ${INPUT} \
--pretrainpath ${PRETRAINPATH}

python neural_model_bert.py eval $MODELPATH \
/bask/projects/j/jlxi8926-auto-sum/waylon/mathbert/tasks/theory-proof-matching/anonymized_dataset/partial_anno_test.csv \
./ranking_eval_output --gpu 0 --max_length 200 --input ${INPUT} \
--pretrainpath ${PRETRAINPATH}

python neural_model_bert.py eval $MODELPATH \
/bask/projects/j/jlxi8926-auto-sum/waylon/mathbert/tasks/theory-proof-matching/anonymized_dataset/adver_anno_test.csv \
./ranking_eval_output --gpu 0 --max_length 200 --input ${INPUT} \
--pretrainpath ${PRETRAINPATH}

python neural_model_bert.py eval $MODELPATH \
/bask/projects/j/jlxi8926-auto-sum/waylon/mathbert/tasks/theory-proof-matching/anonymized_dataset/full_anno_test.csv \
./ranking_eval_output --gpu 0 --max_length 200 --input ${INPUT} \
--pretrainpath ${PRETRAINPATH}

#python neural_model_bert.py eval $MODELPATH \
#/bask/projects/j/jlxi8926-auto-sum/waylon/mathbert/tasks/theory-proof-matching/anonymized_dataset/zero_anno_train.csv \
#. --gpu 0 --max_length 200 --input both \
#--pretrainpath $PRETRAINPATH
#
#python /bask/projects/j/jlxi8926-auto-sum/waylon/mathbert/tasks/theory-proof-matching/maximin_ori/bipartite_matching.py \
#/bask/projects/j/jlxi8926-auto-sum/waylon/mathbert/tasks/theory-proof-matching/maximin_ori/ranks \
#/bask/projects/j/jlxi8926-auto-sum/waylon/mathbert/tasks/theory-proof-matching/maximin_ori/zero_train_ranks_lap