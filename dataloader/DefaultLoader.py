import os
import pickle
import numpy as np

org_path = 'data/'

class DefaultLoader():
    def __init__(self):
        super(DefaultLoader, self).__init__()

    def load_data(self,dataset, validation=True, valid_portion=0.2, maxlen=19, sort_by_len=False):
        data_path = os.path.join(org_path, dataset)
        # Load the dataset
        path_train_data = data_path + '/train.txt'
        path_test_data = data_path + '/test.txt'
        with open(path_train_data, 'rb') as f1:
            train_set = pickle.load(f1)

        with open(path_test_data, 'rb') as f2:
            test_set = pickle.load(f2)

        if validation == False:
            path_valid_data = data_path + '/val.txt'
            with open(path_valid_data, 'rb') as f1:
                valid_set = pickle.load(f1)

        if maxlen:
            new_train_set_x = []
            new_train_set_y = []
            for x, y in zip(train_set[0], train_set[1]):
                if len(x) < maxlen:
                    new_train_set_x.append(x)
                    new_train_set_y.append(y)
                else:
                    # new_train_set_x.append(x[:maxlen])
                    new_train_set_x.append(x[-maxlen:])
                    new_train_set_y.append(y)
            train_set = (new_train_set_x, new_train_set_y)
            del new_train_set_x, new_train_set_y

            new_test_set_x = []
            new_test_set_y = []
            for xx, yy in zip(test_set[0], test_set[1]):
                if len(xx) < maxlen:
                    new_test_set_x.append(xx)
                    new_test_set_y.append(yy)
                else:
                    # new_test_set_x.append(xx[:maxlen])
                    new_test_set_x.append(xx[-maxlen:])
                    new_test_set_y.append(yy)
            test_set = (new_test_set_x, new_test_set_y)
            del new_test_set_x, new_test_set_y

            if validation == False:
                new_valid_set_x = []
                new_valid_set_y = []
                for xx, yy in zip(valid_set[0], valid_set[1]):
                    if len(xx) < maxlen:
                        new_valid_set_x.append(xx)
                        new_valid_set_y.append(yy)
                    else:
                        # new_valid_set_x.append(xx[:maxlen])
                        new_valid_set_x.append(xx[-maxlen:])
                        new_valid_set_y.append(yy)
                valid_set = (new_valid_set_x, new_valid_set_y)
                del new_valid_set_x, new_valid_set_y

            if validation:
                # split training set into validation set
                train_set_x, train_set_y = train_set
                n_samples = len(train_set_x)
                sidx = np.arange(n_samples, dtype='int32')
                np.random.shuffle(sidx)
                n_train = int(np.round(n_samples * (1. - valid_portion)))
                valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
                valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
                train_set_x = [train_set_x[s] for s in sidx[:n_train]]
                train_set_y = [train_set_y[s] for s in sidx[:n_train]]
            else:
                (train_set_x, train_set_y) = train_set
                (valid_set_x, valid_set_y) = valid_set

            (test_set_x, test_set_y) = test_set

            def len_argsort(seq):
                return sorted(range(len(seq)), key=lambda x: len(seq[x]))

            if sort_by_len:
                sorted_index = len_argsort(test_set_x)
                test_set_x = [test_set_x[i] for i in sorted_index]
                test_set_y = [test_set_y[i] for i in sorted_index]

                sorted_index = len_argsort(valid_set_x)
                valid_set_x = [valid_set_x[i] for i in sorted_index]
                valid_set_y = [valid_set_y[i] for i in sorted_index]

            train = (train_set_x, train_set_y)
            valid = (valid_set_x, valid_set_y)
            test = (test_set_x, test_set_y)

            return train, valid, test

    def load_test_data(self,dataset, maxlen=19, sort_by_len=False):
        data_path = os.path.join(org_path, dataset)
        path_test_data = data_path + '/test.txt'
        with open(path_test_data, 'rb') as f2:
            test_set = pickle.load(f2)
        if maxlen:
            new_test_set_x = []
            new_test_set_y = []
            for xx, yy in zip(test_set[0], test_set[1]):
                if len(xx) < maxlen:
                    new_test_set_x.append(xx)
                    new_test_set_y.append(yy)
                else:
                    # new_test_set_x.append(xx[:maxlen])
                    new_test_set_x.append(xx[-maxlen:])
                    new_test_set_y.append(yy)
            test_set = (new_test_set_x, new_test_set_y)
            del new_test_set_x, new_test_set_y

            (test_set_x, test_set_y) = test_set

            def len_argsort(seq):
                return sorted(range(len(seq)), key=lambda x: len(seq[x]))

            if sort_by_len:
                sorted_index = len_argsort(test_set_x)
                test_set_x = [test_set_x[i] for i in sorted_index]
                test_set_y = [test_set_y[i] for i in sorted_index]
            test = (test_set_x, test_set_y)

            return test
