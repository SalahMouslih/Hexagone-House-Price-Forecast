import argparse


def main():
    parser = argparse.ArgumentParser(description='Enter file to preprocesss')
    parser.add_argument('--filename', '-i', type=str, help='Input file')
    args = parser.parse_args()
    
    print('Started pre-processing...')

    processed_file = preprocess(filename=args.filename)

    print('Processed file at', processed_file.path)

if __name__ == "__main__":
    main()
