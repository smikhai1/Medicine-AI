import pandas as pd
import numpy as np
import os
from argparse import ArgumentParser

def get_correct_idx(idx, max_digits):
    idx_string = '0'*max_digits
    i = len(str(idx))
    return idx_string[:-i] + str(idx)

def main():
    parser = ArgumentParser()
    parser.add_argument('-p', help='Path to directory with images', default='./dataset3d/images')
    parser.add_argument('-d', help='Destination for save', default='./dataset3d')

    args = parser.parse_args()
    names = sorted(os.listdir(args.p))

    names = list(map(lambda x: x.split('.')[0]+'.', names))
    names = np.array(names).astype(str).reshape(-1, 1)

    folds = np.zeros((len(names), 1)).astype(int)
    val_idx = np.random.choice(len(names), replace=False, size=int(0.1*len(names)))

    folds[val_idx] = 1

    df = pd.DataFrame(np.hstack((names, folds)), columns=['ImageId', 'fold'])
    df['fold'] = pd.to_numeric(df['fold'])
    df.to_csv(args.d + '/folds.csv', sep='\t')

if __name__ == '__main__':
    main()