import argparse

class Defaultargs():
    def __init__(self):
        super(Defaultargs, self).__init__()

    def get_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--opt_train', type=bool, default=False)
        parser.add_argument('--dataset', default='Tmall',help='Diginetica/Nowplaying/Tmall')
        parser.add_argument('--model', default='CORE', help='NARM/STAMP/SRGNN/AttMix/CORE')
        parser.add_argument('--loss', default='BCE', help='BCE')
        parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
        parser.add_argument('--epoch', type=int, default=100, help='the number of epochs to train for') #100
        parser.add_argument('--batch_size', type=int, default=1024, help='input batch size')  # 512
        parser.add_argument('--embed_dim', type=int, default=64, help='the dimension of item embedding') # 50
        parser.add_argument('--l2', type=float, default=1e-5) # 1e-5
        parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
        parser.add_argument('--lr_dc_step', type=int, default=5, # 5
                            help='the number of steps after which the learning rate decay')
        parser.add_argument('--patience', type=int, default=3, help='the number of epoch to wait before early stop ')
        parser.add_argument('--seed', type=int, default=123, help='the number of epochs to train for') # 123,456,789


        parser.add_argument_group()
        opt = parser.parse_args()
        return opt
