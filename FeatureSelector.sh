#!/bin/bash
figlet "Feature Extractor"
printf "Operation Feature Extraction\n"
INPUT_PCAP=$1
HEADERS_CSV=$2
if [ -z $2 ] 
then
	printf "proceeding without headers\n"
	printf "Getting pcap file\n"
	wc $INPUT_PCAP
	/home/anihilakos/CLionProjects/kdd99_feature_extractor/cmake-build-debug/src/kdd99extractor $INPUT_PCAP >> /home/anihilakos/FeatureExtrFiles/output.csv
	printf "Creating finalOutput.csv at /home/anihilakos/FeatureExtrFiles \n\n" 
	cat /home/anihilakos/FeatureExtrFiles/output.csv > /home/anihilakos/FeatureExtrFiles/finalOutput.csv
	rm /home/anihilakos/FeatureExtrFiles/output.csv
else
	printf "proceeding with headers\n"
	printf "Getting pcap file\n"
	wc $INPUT_PCAP
	printf "Getting Headers \n"
	wc $HEADERS_CSV
	printf "Creating finalOutput.csv at /home/anihilakos/FeatureExtrFiles \n\n" 
	/home/anihilakos/CLionProjects/kdd99_feature_extractor/cmake-build-debug/src/kdd99extractor -e $INPUT_PCAP >> /home/anihilakos/FeatureExtrFiles/output.csv
	cat $HEADERS_CSV /home/anihilakos/FeatureExtrFiles/output.csv > /home/anihilakos/FeatureExtrFiles/finalOutput.csv
	rm /home/anihilakos/FeatureExtrFiles/output.csv
fi








