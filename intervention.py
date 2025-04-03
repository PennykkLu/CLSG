import random
import torch
org_path = '/data/'

class SessionIntervention:
    """
    insert before and in the middle of sequences
    """
    def __init__(self,transforms_book):
        super(SessionIntervention, self).__init__()
        self.transforms_book = transforms_book

    def __call__(self,sess_batch, epochs, n_node, bucketing_ext_len, bucketing_ext_count, all_ext_len):
        # 将在出版后公开
        pass