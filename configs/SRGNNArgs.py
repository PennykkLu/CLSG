import argparse

class SRGNNargs():
    def __init__(self):
        super(SRGNNargs, self).__init__()

    def get_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--validation',type=bool,default=True)
        # parser.add_argument('--valid_portion', type=float, default=0.1,
        #                     help='split the portion of training set as validation set')
        parser.add_argument('--hidden_size', type=int, default=100, help='hidden state size of gru module') #
        parser.add_argument('--num_layers', type=int, default=1) #
        parser.add_argument('--drop_out', type=int, default=0.2) #



        parser.add_argument_group()
        opt = parser.parse_args()
        return opt