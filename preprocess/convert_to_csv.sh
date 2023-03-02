SYM=conservation
DATASETPATH=../datasets

python ./convert_to_csv.py ${DATASETPATH}/${SYM}_train ${DATASETPATH}/${SYM}_train.csv
python ./convert_to_csv.py ${DATASETPATH}/${SYM}_dev ${DATASETPATH}/${SYM}_dev.csv
python ./convert_to_csv.py ${DATASETPATH}/${SYM}_test ${DATASETPATH}/${SYM}_test.csv