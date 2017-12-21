
#Converts .npy to .csv

import argparse
import numpy as np
import os
import pandas as pd


def npy_to_csv(csv_path, npy_path , csv_filename):

    np_array=np.load(npy_path)
    df = pd.DataFrame()
    headers=['Predicted','Actual']
    df=pd.DataFrame(np_array)
    #df.to_csv(csv_path,header=headers,index=None)
    df.to_csv(os.path.join(csv_path,csv_filename),header=headers,index_label='Index /Flow ID')


def parse_args():
    parser = argparse.ArgumentParser(description='Module for converting NPY to CSV files')
    group = parser.add_argument_group('Arguments')
    group.add_argument('-c', '--csv_path', required=True, type=str,
                       help='path of the NPY files to be converted')
    group.add_argument('-n', '--npy_path', required=False, type=str,
                       help='path where converted CSV files will be stored')
    group.add_argument('-f', '--csv_filename', required=False, type=str,
                       help='filename of the CSV file to save')
    arguments = parser.parse_args()
    return arguments


def main(arguments):
    npy_to_csv(arguments.csv_path, arguments.npy_path, arguments.csv_filename)

if __name__ == '__main__':
    args = parse_args()

    main(args)
