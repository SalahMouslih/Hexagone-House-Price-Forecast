import argparse
from data_processing.engine import preprocess_dvf_data
import logging
import os
import warnings
warnings.filterwarnings('ignore')
#from machine_learning.engine import machine_learning_engine
#from eda.engine import eda_engine


# Set up logging
logging.basicConfig(level=logging.INFO)

def parse_file_path(path):
    if os.path.isdir(path):
        # if the path is a directory, return all CSV files in the directory
        return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.csv')]
    else:
        # if the path is a file, return the file path as a list
        return [path]


def main():
    parser = argparse.ArgumentParser(description='Run the machine learning project engines.')

    parser.add_argument('--preprocess', metavar='data_path', nargs='+', type=str,
                        help='path(s) of the raw data file(s) to be preprocessed. You can use wildcards such as * to specify multiple files.')
    parser.add_argument('--ml', dest='ml', action='store_true',
                        help='run the machine learning engine')
    parser.add_argument('--eda', action='store_true',
                        help='run the exploratory data analysis engine')

    args = parser.parse_args()

    if args.preprocess:
        file_paths = [path for pattern in args.preprocess for path in parse_file_path(pattern)]
        logging.info(f"Running the pre-processing engine on files: {', '.join(file_paths)}")
        preprocess_dvf_data(file_paths)
        logging.info("Pre-processing complete. You can find the processed files in the 'processed' directory.")

    #elif args.ml:
     #   logging.info('Running the machine learning engine')
      #  machine_learning_engine()
    #elif args.eda:
     #   logging.info('Running the exploratory data analysis engine')
      #  eda_engine()
    else:
        parser.print_help()

if __name__ == '__main__':
    main()




