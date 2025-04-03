import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_, constant_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class NARM(nn.Module):

    def __init__(self, default_opt,model_opt, n_node):
        super(NARM, self).__init__()
        print("********************")
        print("*****NARMModel*****")
        print("********************")
        self.n_items = n_node
        self.hidden_size = model_opt.hidden_size
        self.batch_size = default_opt.batch_size
        self.n_layers = model_opt.layers
        self.embedding_dim = default_opt.embed_dim
        self.emb = nn.Embedding(self.n_items, self.embedding_dim, padding_idx=0)
        self.emb_dropout = nn.Dropout( model_opt.emb_dropout) # 0.25
        self.gru = nn.GRU(self.embedding_dim, self.hidden_size, self.n_layers,bias=False,batch_first=True)
        self.a_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.a_2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_t = nn.Linear(self.hidden_size, 1, bias=False)
        self.ct_dropout = nn.Dropout(model_opt.ct_dropout) # 0.5
        self.b = nn.Linear(self.embedding_dim, 2 * self.hidden_size, bias=False)
        # self.sf = nn.Softmax()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # parameters initialization
    #     self.apply(self._init_weights)
    #
    # def _init_weights(self, module):
    #     if isinstance(module, nn.Embedding):
    #         xavier_normal_(module.weight.data)
    #     elif isinstance(module, nn.Linear):
    #         xavier_normal_(module.weight.data)
    #         if module.bias is not None:
    #             constant_(module.bias.data, 0)

    def forward(self, seq, lengths):
        seq = seq.transpose(0, 1)
        lengths = lengths.tolist()
        hidden = self.init_hidden(seq.size(1))
        embs = self.emb_dropout(self.emb(seq))
        # embs = self.emb(seq)
        embs = pack_padded_sequence(embs, lengths,enforce_sorted=False)
        gru_out, hidden = self.gru(embs, hidden)
        gru_out, lengths = pad_packed_sequence(gru_out)

        # fetch the last hidden state of last timestamp
        ht = hidden[-1]
        gru_out = gru_out.permute(1, 0, 2)

        c_global = ht
        q1 = self.a_1(gru_out.contiguous().view(-1, self.hidden_size)).view(gru_out.size())
        q2 = self.a_2(ht)

        mask = torch.where(seq.permute(1, 0) > 0, torch.tensor([1.], device=self.device),
                           torch.tensor([0.], device=self.device))
        q2_expand = q2.unsqueeze(1).expand_as(q1)
        q2_masked = mask.unsqueeze(2).expand_as(q1) * q2_expand

        alpha = self.v_t(torch.sigmoid(q1 + q2_masked).view(-1, self.hidden_size)).view(mask.size())
        c_local = torch.sum(alpha.unsqueeze(2).expand_as(gru_out) * gru_out, 1)

        c_t = torch.cat([c_local, c_global], 1)
        c_t = self.ct_dropout(c_t)

        item_embs = self.emb(torch.arange(self.n_items).to(self.device))
        # scores = torch.matmul(c_t, self.b(item_embs).permute(1, 0))
        # scores = self.sf(scores)

        return c_t,self.b(item_embs).permute(1, 0)

    def init_hidden(self, batch_size):
        return torch.zeros((self.n_layers, batch_size, self.hidden_size), requires_grad=True).to(self.device)
