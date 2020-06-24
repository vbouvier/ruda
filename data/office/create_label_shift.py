import numpy as np

datasets = ['amazon_list.txt', 'webcam_list.txt', 'dslr_list.txt']

N = 5


def get_label_from_line(line):
    return eval(line.split()[-1])


def get_filename(file):
    return file[:-4] + '_shift_' + str(N) + '.txt'


for dataset in datasets:
    with open(dataset, 'r') as f:
        with open(get_filename(dataset), 'w') as new_f:
            for line in f:
                label = get_label_from_line(line)
                if label > 15:
                    for _ in range(N):
                        new_f.write(line)
                else:
                    new_f.write(line)