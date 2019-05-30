from argparse import ArgumentParser
from transform_dataset import transform_3d_data

def main():
    parser = ArgumentParser()
    parser.add_argument('source', help='path to the original dataset root directory')
    parser.add_argument('dest', help='path to the root directory of transformed dataset')
    args = parser.parse_args()

    transform_3d_data(args.source, args.dest)
    print('Dataset transformed successfully!', end='\n\n')

if __name__ == '__main__':
    main()