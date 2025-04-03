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
    # 将在出版后公开
    pass


if __name__ == "__main__":
    get_popularity(pickle.load(open(data_path + '/unprefixed_train_data.txt', 'rb'))[0])
    get_dict_pre_popularity()