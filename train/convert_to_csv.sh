SYM=conservation
DATASETPATH=../datasets/unmixed

python ../train/convert_to_csv.py ${DATASETPATH}/${SYM}_train ${DATASETPATH}/${SYM}_train.csv
python ../train/convert_to_csv.py ${DATASETPATH}/${SYM}_dev ${DATASETPATH}/${SYM}_dev.csv
python ../train/convert_to_csv.py ${DATASETPATH}/${SYM}_test ${DATASETPATH}/${SYM}_test.csv