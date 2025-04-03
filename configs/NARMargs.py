import argparse

class NARMargs():
    def __init__(self):
        super(NARMargs, self).__init__()

    def get_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--hidden_size', type=int, default=100, help='hidden state size of gru module') #
        parser.add_argument('--feat_size', type=int, default= 2 * 100, help='hidden size of output feature') #
        parser.add_argument('--layers', type=int, default=1, help='the number of model layers ') # 50
        parser.add_argument('--emb_dropout', default=0.1, type=float)
        parser.add_argument('--ct_dropout', default=0.1, type=float)

        parser.add_argument_group()
        opt = parser.parse_args()
        return opt