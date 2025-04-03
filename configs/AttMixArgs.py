import argparse

class AttMixargs():
    def __init__(self):
        super(AttMixargs, self).__init__()

    def get_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--validation',type=bool,default=True)
        # parser.add_argument('--valid_portion', type=float, default=0.1,
        #                     help='split the portion of training set as validation set')
        parser.add_argument('--hidden_size', type=int, default=256, help='hidden state size of gru module') #
        parser.add_argument('--alpha', type=float, default=0.75, help='parameter for beta distribution')
        parser.add_argument('--heads', type=int, default=8, help='number of attention heads')
        parser.add_argument('--use_lp_pool', type=str, default="True")
        parser.add_argument('--train_flag', type=str, default="True")
        parser.add_argument('--PATH', default='../checkpoint/Atten-Mixer_diginetica.pt', help='checkpoint path')
        parser.add_argument('--softmax', type=bool, default=True)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--dot', default=True, action='store_true')
        parser.add_argument('--last_k', type=int, default=7)
        parser.add_argument('--l_p', type=int, default=4)
        parser.add_argument('--norm', default=True, help='adapt NISER, l2 norm over item and session embedding')

        parser.add_argument_group()
        opt = parser.parse_args()
        return opt