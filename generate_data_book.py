import pickle
import math
from collections import Counter

org_path = '/data/'
data_path = org_path + 'Diginetica' # Nowplaying Tmall

def get_popularity(data):
    """
    :return:  dict  key: item  value: be interacted times
    """
    all_elements = []
    for sess in data:
        all_elements.extend(sess)
    element_counts = Counter(all_elements)

    with open(data_path+'/Dict_item_popularity.pkl','wb') as f:
        pickle.dump(element_counts,f)
    return dict(element_counts)


def get_dict_pre_popularity():
    """
    support = sum_all(1/(distances * log(1+N(i))))
    :return:   key: anchor  value:[{preID:support},{...}]
    """
    dict_item_pre = {}
    train_data = pickle.load(open(data_path + '/unprefixed_train_data.txt', 'rb'))[0]
    dict_popularity = get_popularity(train_data)
    for sess in train_data:
        max_idx = len(sess)
        if max_idx == 1:
            continue
        for anchor_idx in range(max_idx):
            anchor_item = sess[anchor_idx]
            if anchor_item not in dict_item_pre.keys():
                dict_item_pre[anchor_item] = {}
            for pre_idx in range(anchor_idx):
                pre_item = sess[pre_idx]
                if pre_item not in dict_item_pre[anchor_item].keys():
                    dict_item_pre[anchor_item][pre_item] = 0
                cooccurrence_weight = 1 / (anchor_idx-pre_idx)
                popularity_weight = 1 / math.log(1 + dict_popularity[anchor_item])
                dict_item_pre[anchor_item][pre_item] += cooccurrence_weight * popularity_weight
    with open(data_path+'/Dict_item_pre_popularity.pkl','wb') as f:
        pickle.dump(dict_item_pre,f)


if __name__ == "__main__":
    get_popularity(pickle.load(open(data_path + '/unprefixed_train_data.txt', 'rb'))[0])
    get_dict_pre_popularity()
