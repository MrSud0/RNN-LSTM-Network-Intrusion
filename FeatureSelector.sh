#!/bin/bash
#this scripts runs a feature extractor located in my machine and outputs a csv file with all the features extracted ,shape[samples,features]

figlet "Feature Extractor"

printf "Operation Feature Extraction\n"
printf "Getting pcap file \n"
INPUT_PCAP=$1
wc $INPUT_PCAP

printf "Getting Headers \n"
HEADERS_CSV=$2
wc $HEADERS_CSV

printf "Creating finalOutput.csv at /home/anihilakos/FeatureExtrFiles \n\n"
/home/anihilakos/CLionProjects/kdd99_feature_extractor/cmake-build-debug/src/kdd99extractor -e $INPUT_PCAP >> /home/anihilakos/FeatureExtrFiles/output.csv
rm output.csv
cat $HEADERS_CSV /home/anihilakos/FeatureExtrFiles/outputpart1.csv > /home/anihilakos/FeatureExtrFiles/finalOutput.csv