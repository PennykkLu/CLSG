import argparse

class STAMPargs():
    def __init__(self):
        super(STAMPargs, self).__init__()

    def get_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--validation',type=bool,default=True)
        parser.add_argument('--valid_portion', type=float, default=0.1,
                            help='split the portion of training set as validation set')
        parser.add_argument('--hidden_size', type=int, default=100)  #
        parser.add_argument('--dropout_prob', default=0.4, type=float)

        parser.add_argument_group()
        opt = parser.parse_args()
        return opt