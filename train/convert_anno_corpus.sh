ANNO=conservation
python ./corpus_utils.py ./rawdata ./${ANNO} --preprocessing f
python ./convert_to_csv.py ./${ANNO}_train ./${ANNO}_train.csv
python ./convert_to_csv.py ./${ANNO}_dev ./${ANNO}_dev.csv
python ./convert_to_csv.py ./${ANNO}_test ./${ANNO}_test.csv