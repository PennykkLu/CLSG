
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import dataloader


class COREData():
    def __init__(self,dataset,batch_size,validation=True):
        self.dataset = dataset
        self.validation = validation
        self.batch_size = batch_size
        self.loader = getattr(dataloader, 'DefaultLoader')()

    def get_loader(self,opt_train):
        if opt_train:
            train, valid, test = self.loader.load_data(self.dataset, validation=self.validation)
            train_data = MyDataset(train)
            valid_data = MyDataset(valid)
            test_data = MyDataset(test)
            train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)
            valid_loader = DataLoader(valid_data, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)
            test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)
            return train_loader,valid_loader,test_loader
        else:
            test = self.loader.load_test_data(self.dataset)
            test_data = MyDataset(test)
            test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)
            return test_loader


def collate_fn(data):
    data.sort(key=lambda x: len(x[0]), reverse=True)
    lens = [len(sess) for sess, label in data]
    labels = []
    padded_sess = torch.zeros(len(data), max(lens)).long()
    for i, (sess_org, label) in enumerate(data):
        padded_sess[i, :lens[i]] = torch.LongTensor(sess_org)
        labels.append(label)

    return torch.tensor(labels).long(), padded_sess,torch.tensor(lens).long()


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
        print('-' * 50)
        print('Dataset info:')
        print('Number of sessions: {}'.format(len(data[0])))
        print('-' * 50)

    def __getitem__(self, index):
        session_items = self.data[0][index]
        target = self.data[1][index]
        return session_items,target

    def __len__(self):
        return len(self.data[0])

