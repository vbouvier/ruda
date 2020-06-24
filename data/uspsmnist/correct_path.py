paths = ['mnist_train.txt', 'mnist_test.txt', 'usps_train.txt', 'usps_test.txt']

for path in paths:

    with open(path, 'r') as f:
        lines = f.readlines()

    with open(path[:-4] + '_local.txt', 'w') as f:
        for line in lines:
            print(line)
            f.write('../data/' + line[33:])