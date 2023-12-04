from helper import *
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class TrainDataset(Dataset):
    """
    Training Dataset class.

    Parameters
    ----------
    triples:    The triples used for training the model
    params:     Parameters for the experiments

    Returns
    -------
    A training Dataset class instance used by DataLoader
    """

    def __init__(self, triples, params):
        self.triples = triples
        self.p = params
        self.entities = np.arange(self.p.num_ent, dtype=np.int32)

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        ele = self.triples[idx]
        triple = torch.LongTensor(ele['triple'])

        trp_label = torch.LongTensor(ele['label'])

        text_ids, text_mask = torch.LongTensor(ele['text_ids']), torch.LongTensor(ele['text_mask'])
        pred_pos = torch.LongTensor([ele['pred_pos']])

        # if self.p.lbl_smooth != 0.0:
        #     trp_label = (1.0 - self.p.lbl_smooth) * trp_label + (1.0 / self.p.num_ent)

        return triple, trp_label, text_ids, text_mask, pred_pos



    @staticmethod
    def collate_fn(data):
        triple = torch.stack([_[0] for _ in data], dim=0)
        trp_label = torch.stack([_[1] for _ in data], dim=0)
        # triple: (batch-size) * 3(sub, rel, -1) trp_label (batch-size) * num entity
        # return triple, trp_label
        text_ids = pad_sequence([_[2] for _ in data], batch_first=True, padding_value=0)
        text_mask = pad_sequence([_[3] for _ in data], batch_first=True, padding_value=0)
        pred_pos = torch.stack([_[4] for _ in data], dim=0)

        return triple, trp_label.squeeze(-1), text_ids, text_mask, pred_pos




    def get_label(self, label):
        y = np.zeros([self.p.num_ent], dtype=np.float32)
        for e2 in label: y[e2] = 1.0
        return torch.FloatTensor(y)



class TestDataset(Dataset):
    """
    Evaluation Dataset class.

    Parameters
    ----------
    triples:    The triples used for evaluating the model
    params:     Parameters for the experiments

    Returns
    -------
    An evaluation Dataset class instance used by DataLoader for model evaluation
    """

    def __init__(self, triples, params):
        self.triples = triples
        self.p = params

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        ele = self.triples[idx]
        triple, label = torch.LongTensor(ele['triple']), np.int32(ele['label'])
        label = self.get_label(label)

        text_ids, text_mask = torch.LongTensor(ele['text_ids']), torch.LongTensor(ele['text_mask'])
        pred_pos = torch.LongTensor([ele['pred_pos']])
        return triple, label, text_ids, text_mask, pred_pos

    @staticmethod
    def collate_fn(data):
        triple = torch.stack([_[0] for _ in data], dim=0)
        label = torch.stack([_[1] for _ in data], dim=0)
        text_ids = pad_sequence([_[2] for _ in data], batch_first=True, padding_value=0)
        text_mask = pad_sequence([_[3] for _ in data], batch_first=True, padding_value=0)
        pred_pos = torch.stack([_[4] for _ in data], dim=0)

        return triple, label, text_ids, text_mask, pred_pos

    def get_label(self, label):
        y = np.zeros([self.p.num_ent], dtype=np.float32)
        for e2 in label: y[e2] = 1.0
        return torch.FloatTensor(y)
