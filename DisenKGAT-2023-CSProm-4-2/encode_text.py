from helper import *
from data_loader import *
from model import *
import transformers
from transformers import AutoConfig, BertTokenizer, RobertaTokenizer, AutoModel
transformers.logging.set_verbosity_error()
from tqdm import tqdm


def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True


# sys.path.append('./')

class TrainDataset(Dataset):


    def __init__(self, entities, params):
        self.entities = entities
        self.p = params

    def __len__(self):
        return len(self.entities)

    def __getitem__(self, idx):
        ele = self.entities[idx]
        # ent = torch.LongTensor(ele['ent_id'])

        ent = torch.FloatTensor([np.float(ele['ent_id'])])


        text_ids, text_mask = torch.LongTensor(ele['input_ids']), torch.LongTensor(ele['input_mask'])


        return ent, text_ids, text_mask



    @staticmethod
    def collate_fn(data):
        ent = torch.cat([_[0] for _ in data], dim=0)
        text_ids = pad_sequence([_[1] for _ in data], batch_first=True, padding_value=0)
        text_mask = pad_sequence([_[2] for _ in data], batch_first=True, padding_value=0)

        return ent, text_ids, text_mask




def load_data(p):

    def read_file2(data_path, filename):
        items = []
        file_name = os.path.join(data_path, filename)
        with open(file_name, encoding='utf-8') as file:
            lines = file.read().strip('\n').split('\n')
        for i in range(1, len(lines)):
            item, id = lines[i].strip().split('\t')
            items.append(item.lower())
        return items


    data_path = os.path.join('../data', p.dataset)
    # load ids of CSProm
    entities = read_file2(data_path, 'entity2id.txt')

    ent2id = {ent: idx for idx, ent in enumerate(entities)}

    id2ent = {idx: ent for ent, idx in ent2id.items()}


    # load textual data
    ent_names = read_file('../data', p.dataset, 'entityid2name.txt', 'name')
    ent_descs = read_file('../data', p.dataset, 'entityid2description.txt', 'desc')
    p.pretrained_model_name = p.pretrained_model.split('/')[-1].split('_')[0]
    if p.pretrained_model_name.lower() == 'bert':
        tok = BertTokenizer.from_pretrained(p.pretrained_model, add_prefix_space=False)
    elif p.pretrained_model_name.lower() == 'roberta':
        tok = RobertaTokenizer.from_pretrained(p.pretrained_model, add_prefix_space=False)

    triples_save_file = '../data/{}/{}_{}.txt'.format(p.dataset, p.pretrained_model_name.lower(), 'entity_tokens')

    if os.path.exists(triples_save_file):
        entity_cons = json.load(open(triples_save_file))
    else:
        entity_cons = []
        for ent in id2ent.keys():
            sub_name = ent_names[ent]
            sub_desc = ent_descs[ent]
            sub_text = sub_name + ' ' + sub_desc

            tokenized_text = tok(sub_text, max_length=p.text_len, truncation=True)
            input_ids = tokenized_text.input_ids  # encoded tokens
            input_mask = tokenized_text.attention_mask  # the position of padding is zero, others are one
            # print(self.tok.decode(source_ids))
            # print(source_ids)
            entity_cons.append({'ent_id': ent, 'input_ids': input_ids, 'input_mask': input_mask})

        json.dump(entity_cons, open(triples_save_file, 'w'))

    # print(type(entity_cons))
    return entity_cons, len(ent2id)






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser For Arguments',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # FB15k-237
    parser.add_argument('-data', dest='dataset', default='FB15k-237', help='Dataset to use, default: FB15k-237')

    parser.add_argument('-gpu', type=int, default=6, help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')

    ## LLM params
    parser.add_argument('-pretrained_model', type=str, default='bert_large', choices = ['bert_large', 'bert_base', 'roberta_large', 'roberta_base'])
    parser.add_argument('-pretrained_model_name', type=str, default='bert', help='')
    parser.add_argument('-text_len', default=72, type=int, help='')
    parser.add_argument('-desc_max_length', default=40, type=int, help='')
    parser.add_argument('-num_workers', type=int, default=0, help='Number of processes to construct batches')
    parser.add_argument('-batch', default=20, type=int, help='Batch size')
    args = parser.parse_args()

    if args.gpu != '-1' and torch.cuda.is_available():
        device = torch.device('cuda')
        if args.gpu < torch.cuda.device_count():
            torch.cuda.set_device(args.gpu)
        torch.cuda.set_rng_state(torch.cuda.get_rng_state())
        torch.backends.cudnn.deterministic = True
    else:
        device = torch.device('cpu')
    args.pretrained_model_name = args.pretrained_model.split('_')[0]
    
    if args.pretrained_model == 'bert_large':
        args.pretrained_model = '/home/zjlab/gengyx/LMs/BERT_large'
    elif args.pretrained_model == 'bert_base':
        args.pretrained_model = '/home/zjlab/gengyx/LMs/BERT_base'
    elif args.pretrained_model == 'roberta_large':
        args.pretrained_model = '/home/zjlab/gengyx/LMs/RoBERTa_large'
    elif args.pretrained_model == 'roberta_base':
        args.pretrained_model = '/home/zjlab/gengyx/LMs/RoBERTa_base'


    args.vocab_size = AutoConfig.from_pretrained(args.pretrained_model).vocab_size
    args.model_dim = AutoConfig.from_pretrained(args.pretrained_model).hidden_size
    lm_config = AutoConfig.from_pretrained(args.pretrained_model)
    lm_model = AutoModel.from_pretrained(args.pretrained_model, config=lm_config).to(device)


    for p in lm_model.parameters():
        p.requires_grad = False

    entity_cons, ent_num = load_data(args)

    data_iter = DataLoader(
        TrainDataset(entity_cons, args),
        batch_size=args.batch,
        shuffle=True,
        num_workers=max(0, args.num_workers),
        collate_fn=TrainDataset.collate_fn
    )

    train_iter = iter(data_iter)


    def read_batch(batch):
        ent, text_ids, text_mask = [_.to(device) for _ in batch]
        return ent, text_ids, text_mask

    text_embeds = torch.zeros([ent_num, args.model_dim], dtype=torch.float)
    for step, batch in tqdm(enumerate(train_iter)):
        ent, text_ids, text_mask = read_batch(batch)
        # print(text_ids)
        out = lm_model(input_ids=text_ids, attention_mask=text_mask)

        last_hidden_state = out.last_hidden_state
        # print(last_hidden_state)
        sent = torch.mean(last_hidden_state, dim=1)
        # sent = last_hidden_state

        # print(sent.shape)
        for i in range(ent.size(0)):
            ent_id = int(ent[i])
            text_embeds[ent_id] = sent[i].detach().cpu()
            # text_embeds[ent_id] = sent[i][0].detach().cpu()

    embeds_save_file = '../data/{}/entity_embeds_{}.pt'.format(args.dataset, args.pretrained_model.split('/')[-1].lower())
    torch.save(text_embeds, embeds_save_file)

    # print(text_embeds)