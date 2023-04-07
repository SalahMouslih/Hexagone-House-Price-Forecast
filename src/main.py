"""
This module contains the main function that runs engines on a set of files.
The pre-processing engine performs several transformations on the input files to
prepare them for further processing.
The machine learnibg engine..
"""

import argparse
import logging
import traceback
import os
import warnings
warnings.filterwarnings('ignore')

from data_processing.engine import preprocessing_engine
from machine_learning.engine import ml_engine
from eda.eda_engine import eda_engine


# Set up logging
logging.basicConfig(level=logging.INFO)

def parse_file_path(path):
    """ Parse file paths """
    if not os.path.exists(path):
        # raise an error if the path does not exist
        raise FileNotFoundError(f"No such file or directory: '{path}'")
    if os.path.isdir(path):
        # if the path is a directory, return all CSV files in the directory
        return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.csv')]
    if os.path.isfile(path):
        # if the path is a file, return the file path as a list
        return [path]
    raise ValueError(f"Invalid input: '{path}' is not a file or directory")


def main():
    """ main function """
    parser = argparse.ArgumentParser(description='Run the machine learning project engines.')

    parser.add_argument('--preprocess', metavar='data_path', nargs='+', type=str,
                        help='path(s) of the raw data file(s) to be preprocessed.'
                        'You can use wildcards such as * to specify multiple files.')
    parser.add_argument('--ml', dest='ml', action='store_true',
                        help='run the machine learning engine')
    parser.add_argument('--eda', metavar='data_path', type=str,
                        help='run the exploratory data analysis engine')

    args = parser.parse_args()

    if args.preprocess:
        try:
            file_paths = [path for pattern in args.preprocess for path in parse_file_path(pattern)]
            logging.info("Running the pre-processing engine on files: %s", ', '.join(file_paths))
            if preprocessing_engine(file_paths):
                logging.info("Pre-processing completed successfully!")
            else:
                logging.error("Error occurred during pre-processing!")
        except Exception as e:
            logging.error("An error occurred while running the pre-processing engine: %s", e)
            print(traceback.format_exc())
    elif args.eda:
        logging.info('Running the exploratory data analysis engine')
        if eda_engine(args.eda):
            logging.info("EDA complete. You can find the output plots "
             "in the 'plots' directory.")
        else:
            logging.error("An error occurred while performing EDA")
    elif args.ml:
        logging.info('Running the machine learning engine')
        ml_engine()
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
