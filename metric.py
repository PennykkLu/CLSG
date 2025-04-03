import numpy as np


def metric(scores, targets, k=20):
    sub_scores = scores.topk(k)[1].cpu().detach().numpy()
    targets = targets.cpu().detach().numpy()
    res = []
    for score,target in zip(sub_scores,targets):
        hit = float(np.isin(target, score))
        if len(np.where(score == target)[0]) == 0:
            mrr = 0
            ndcg = 0
        else:
            rank = np.where(score == target)[0][0] + 1
            mrr = 1 / rank
            ndcg = 1 / np.log2(rank + 1)
        res.append([hit,mrr,ndcg])

    return res


def bucketing_metric(scores,targets,bucketing_basis,bucketing_hit5,bucketing_mrr5,bucketing_hit10,bucketing_mrr10,bucketing_hit20,bucketing_mrr20,bucketing_count):
    targets = targets.cpu().detach().numpy()
    for k in [5,10,20]:
        sub_scores = scores.topk(k)[1].cpu().detach().numpy()
        for score,target,basis in zip(sub_scores,targets,bucketing_basis):
            hit = float(np.isin(target, score))
            basis = int(basis)
            if len(np.where(score == target)[0]) == 0:
                mrr = 0
            else:
                rank = np.where(score == target)[0][0] + 1
                mrr = 1 / rank
            if k == 5:
                bucketing_hit5[basis] += hit
                bucketing_mrr5[basis] += mrr
                bucketing_count[basis] += 1
            elif k == 10:
                bucketing_hit10[basis] += hit
                bucketing_mrr10[basis] += mrr
            elif k == 20:
                bucketing_hit20[basis] += hit
                bucketing_mrr20[basis] += mrr
    return bucketing_hit5,bucketing_mrr5,bucketing_hit10,bucketing_mrr10,bucketing_hit20,bucketing_mrr20,bucketing_count


