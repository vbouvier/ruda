import numpy as np

datasets = ['mnist_train.txt', 'usps_train.txt']


def get_label_from_line(line):
    return eval(line.split()[-1])


def get_filename(file):
    return file[:-4] + '_shift_25.txt'

for dataset in datasets:
    with open(dataset, 'r') as f:
        with open(get_filename(dataset), 'w') as new_f:
            for line in f:
                label = get_label_from_line(line)
                if label < 6:
                    if np.random.rand() < 0.25:
                        new_f.write(line)
                else:
                    new_f.write(line)