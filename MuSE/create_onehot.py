import pandas as pd
import numpy as np

def seq_to_hot(filename):
    base_map = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'C': [0, 0, 1, 0], 'G': [0, 0, 0, 1]}
    df = pd.read_csv(f'datasets/{filename}.csv')
    seqs = df['sequence'].values
    label = df['label'].values
    seqs_hot = []
    for seq in seqs:
        seq_hot_temp = []
        for base in seq:
            if base == 'N':
                seq_hot_temp.append([0, 0, 0, 0])
            else:
                seq_hot_temp.append(base_map[base])
        if len(seq_hot_temp) <= 3000:
            for i in range(3000 - len(seq_hot_temp)):
                seq_hot_temp.append([0, 0, 0, 0])
        seqs_hot.append(seq_hot_temp)
    seqs_hot = np.array(seqs_hot)
    np.savez(f'datasets/one-hot/{filename}.npz', seq=seqs_hot, label=label)

if __name__ == '__main__':
    names = ['train_mouse', 'test_mouse', 'train_human', 'test_human']
    for name in names:
        seq_to_hot(name)