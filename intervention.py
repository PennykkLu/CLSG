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
        batch_size, MAX_LEN = sess_batch.shape
        new_sess_batch = torch.zeros(batch_size, MAX_LEN).long()
        for i, session in enumerate(sess_batch):
            org_len = int(torch.count_nonzero(session))
            new_session = session[:org_len].tolist()
            for _ in range(epochs):
                idx = 0
                while len(new_session) < MAX_LEN and idx < len(new_session):
                    if random_flag:
                        """
                        random insert
                        """
                        random_item = random.randint(0, n_node - 1)
                        new_session = new_session[:idx] + [random_item] + new_session[idx:]
                        idx += 2
                    else:
                        """
                        guided insert
                        """
                        anchor_item = new_session[idx]
                        if anchor_item not in self.transforms_book.keys() or len(list(self.transforms_book[anchor_item])) == 0: # no match
                            idx += 1
                            continue
                        sorted_dict = dict(sorted(self.transforms_book[anchor_item].items(), key=lambda item: item[1], reverse=True))
                        if idx == 0: # the first item
                            for pre_item in sorted_dict.keys():
                                # if sorted_dict[pre_item] < self.threshold_conf:
                                #     break
                                if pre_item == anchor_item:
                                    continue
                                new_session = [pre_item] + new_session
                                idx += 1
                                break
                            idx += 1
                            continue
                        for pre_item in sorted_dict.keys(): # the middle item
                            # if sorted_dict[pre_item] < self.threshold_conf:
                            #     break
                            if pre_item == anchor_item \
                                    or pre_item not in self.transforms_book.keys() \
                                    or new_session[idx-1] not in self.transforms_book[pre_item]\
                                    or pre_item in new_session:
                                continue
                            new_session = new_session[:idx] + [pre_item] + new_session[idx:]
                            idx += 1
                            break
                        idx += 1

            new_session = new_session + [0] * max(MAX_LEN - len(new_session), 0)
            new_sess_batch[i, :] = torch.LongTensor(new_session)
            new_len = int(torch.count_nonzero(new_sess_batch[i, :]))
            all_ext_len.append(new_len)
            bucketing_ext_len[org_len] += new_len
            bucketing_ext_count[org_len] += 1
        new_lens_batch = torch.count_nonzero(new_sess_batch, dim=1)

        return new_sess_batch, new_lens_batch, bucketing_ext_len, bucketing_ext_count,all_ext_len
