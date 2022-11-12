import torch


class NegativeSampler:
    def __init__(self, dataset):
        self.dataset = dataset
    
    def sampling(self, head: int, rel: int, tail: int, num_negs: int, **kwargs) -> torch.Tensor:
        """Generate negatives for one triplet.

        Args:
            head: head entity id
            rel: relation id
            tail: tail entity id
            num_negs: number of negatives to generate
        """
        raise NotImplementedError
    
    def _corrupt_head(self, tail: int, rel: int, num_max: int):
        neg = torch.randint(low=0, high=self.dataset.num_ents, size=(num_max, 1))
        mask = torch.isin(neg, torch.Tensor(list(self.dataset.h_of_tr[(tail, rel)])),
                          assume_unique=True, invert=True)
        neg = neg[mask]
        return neg

    def _corrupt_tail(self, head: int, rel: int, num_max: int):
        neg = torch.randint(low=0, high=self.dataset.num_ents, size=(num_max, 1))
        mask = torch.isin(neg, torch.Tensor(list(self.dataset.t_of_hr[(head, rel)])),
                          assume_unique=True, invert=True)
        neg = neg[mask]
        return neg


class UniformNegativeSampler(NegativeSampler):
    def __init__(self, dataset):
        self.dataset = dataset
        
    def sampling(self, head: int, rel: int, tail: int, num_negs: int) -> torch.Tensor:
        num_h_negs = num_negs // 2 if num_negs > 1 else 1
        num_t_negs = num_negs // 2 if num_negs > 1 else 1
        
        neg_list_h = []
        while len(neg_list_h) < num_h_negs:
            neg_tmp_h = self._corrupt_head(tail, rel, num_max=(num_h_negs - len(neg_list_h)) * 2)
            if len(neg_tmp_h):
                neg_list_h.append(neg_tmp_h)
        neg_list_h = torch.vstack(neg_list_h).reshape(1, -1)  # (1, num_negs)
        # print(neg_list_h)

        neg_list_t = []
        while len(neg_list_t) < num_t_negs:
            neg_tmp_t = self._corrupt_tail(head, rel, num_max=(num_t_negs - len(neg_list_t)) * 2)
            if len(neg_tmp_t):
                neg_list_t.append(neg_tmp_t)
        neg_list_t = torch.vstack(neg_list_t).reshape(1, -1)

        return neg_list_h[:, :num_h_negs], neg_list_t[:, :num_t_negs]
    