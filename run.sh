#!/bin/bash
figlet  "Intrusion Detection System"
figlet -f small.flf "POWERED BY LSTM"
printf "Available Operations: \\n[1] Train \\n[2] Test \\n[3] Results \\n[4] NPY Converter \\n[5] Feature Extractor\n>> "
source activate keras_env
read choice
if [ "$choice" -eq "1" ] &>/dev/null; then
	echo "Training LSTM model for intrusion detection"
	if [ ! -d "checkpoints/weights" ] &>/dev/null; then
		mkdir models/checkpoint_path
	fi
	python3 LSTM_main.py --operation "train" \
    --train_dataset datasets/KDDTrainFinal.csv \
    --checkpoint_path checkpoints/weights \
    --save_model modelSaves/  \
    --result_path results/training
elif [ "$choice" -eq "2" ] &>/dev/null; then
	if [ -d "modelSaves" ] &>/dev/null; then
		echo "Testing LSTM model for intrusion detection"
		python3 LSTM_main.py --operation "test" \
        --test_dataset datasets/KDDTest.csv \
        --load_model modelSaves/model_high.h5 \
        --result_path results/testing
	else
		echo "Train the model first!"
		exit
	fi
elif [ "$choice" -eq "3" ] &>/dev/null; then
	python3 results.py -v results/testing 
elif [ "$choice" -eq "4" ] &>/dev/null; then
	python3 utils/npy_to_csv.py -c results/CSVs -n results/testing/testing-LSTM-Results-0.85.npy -f results.csv
elif [ "$choice" -eq "5" ] &>/dev/null; then
	./FeatureExtractor.sh input.pcap headers.txt
else
	echo "Invalid input"
	exit
fi
