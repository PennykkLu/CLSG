import argparse

class COREargs():
    def __init__(self):
        super(COREargs, self).__init__()

    def get_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--validation',type=bool,default=True)
        parser.add_argument('--embedding_size', type=int, default=64) #
        parser.add_argument('--inner_size', type=int, default=256) #
        parser.add_argument('--n_layers', type=int, default=1) #
        parser.add_argument('--n_heads', type=int, default=8) #
        parser.add_argument('--hidden_dropout_prob', type=float, default=0.7) #
        parser.add_argument('--attn_dropout_prob', type=float, default=0.5) #
        parser.add_argument('--hidden_act', default='gelu') #
        parser.add_argument('--layer_norm_eps', type=float, default=1e-12)  #
        parser.add_argument('--initializer_range', type=float, default=0.02)  #
        parser.add_argument('--sess_dropout', type=float, default=0.2)  #
        parser.add_argument('--item_dropout', type=float, default=0.2)  #
        parser.add_argument('--temperature', type=float, default=0.07)  #

        parser.add_argument_group()
        opt = parser.parse_args()
        return opt