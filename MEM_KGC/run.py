from helper import *
from data_loader import *
from model import *
import transformers
from transformers import AutoConfig, BertTokenizer
transformers.logging.set_verbosity_error()
from tqdm import tqdm
from transformers import AdamW

def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True


# sys.path.append('./')

class Runner(object):

    def load_data(self):
        def construct_input_text(sub, rel):
            sub_name = self.ent_names[sub]
            sub_desc = self.ent_descs[sub]
            sub_text = sub_name + ' ' + sub_desc
            if rel < self.p.num_rel:
                rel_text = self.rel_names[rel]
            else:
                rel_text = 'reversed: ' + self.rel_names[rel - self.p.num_rel]
            tokenized_text = self.tok(sub_text, text_pair=rel_text, max_length=self.p.text_len-1, truncation=True)
            source_ids = tokenized_text.input_ids  # encoded tokens
            source_mask = tokenized_text.attention_mask  # the position of padding is zero, others are one
            # (manually) insert [MASK] token at the end of this sequence
            source_ids.insert(-1, 103)
            source_mask.insert(-1, 1)
            return source_ids, source_mask
        def read_file2(data_path, filename):
            items = []
            file_name = os.path.join(data_path, filename)
            with open(file_name, encoding='utf-8') as file:
                lines = file.read().strip('\n').split('\n')
            for i in range(1, len(lines)):
                item, id = lines[i].strip().split('\t')
                items.append(item.lower())
            return items


        data_path = os.path.join('../data', self.p.dataset)
        # load ids of CSProm
        entities = read_file2(data_path, 'entity2id.txt')
        relations = read_file2(data_path, 'relation2id.txt')

        self.ent2id = {ent: idx for idx, ent in enumerate(entities)}
        self.rel2id = {rel: idx for idx, rel in enumerate(relations)}
        self.rel2id.update({rel + '_reverse': idx + len(self.rel2id) for idx, rel in enumerate(relations)})

        self.id2ent = {idx: ent for ent, idx in self.ent2id.items()}
        self.id2rel = {idx: rel for rel, idx in self.rel2id.items()}

        self.p.num_ent = len(self.ent2id)
        self.p.num_rel = len(self.rel2id) // 2
        # self.p.embed_dim = self.p.k_w * self.p.k_h if self.p.embed_dim is None else self.p.embed_dim
        print('num_ent {} num_rel {}'.format(self.p.num_ent, self.p.num_rel))
        self.data = ddict(list)
        sr2o = ddict(set)

        # self.text = dict()
        for split in ['train', 'test', 'valid']:
            for line in open('../data/{}/{}.txt'.format(self.p.dataset, split)):
                sub, rel, obj = map(str.lower, line.strip().split('\t'))
                sub, rel, obj = self.ent2id[sub], self.rel2id[rel], self.ent2id[obj]
                self.data[split].append((sub, rel, obj))

                if split == 'train':
                    sr2o[(sub, rel)].add(obj)
                    sr2o[(obj, rel + self.p.num_rel)].add(sub)


        # self.data: all origin train + valid + test triplets
        self.data = dict(self.data)
        # print(self.data['test'][:20])
        # self.sr2o: train origin edges and reverse edges
        self.sr2o = {k: list(v) for k, v in sr2o.items()}
        for split in ['test', 'valid']:
            for sub, rel, obj in self.data[split]:
                sr2o[(sub, rel)].add(obj)
                sr2o[(obj, rel + self.p.num_rel)].add(sub)

        self.sr2o_all = {k: list(v) for k, v in sr2o.items()}

        # load textual data
        self.ent_names = read_file('../data', self.p.dataset, 'entityid2name.txt', 'name')
        self.rel_names = read_file('../data', self.p.dataset, 'relationid2name.txt', 'name')
        self.ent_descs = read_file('../data', self.p.dataset, 'entityid2description.txt', 'desc')
        self.tok = BertTokenizer.from_pretrained(self.p.pretrained_model, add_prefix_space=False)


        # for quickly load data
        triples_save_file = '../data/{}/{}.txt'.format(self.p.dataset, 'loaded_triples_bert')
        if os.path.exists(triples_save_file):
            self.triples = json.load(open(triples_save_file))
        else:
            self.triples = ddict(list)
            for sub, rel, obj in tqdm(self.data['train']):
                text_ids, text_mask = construct_input_text(sub, rel)
                self.triples['train'].append({'triple': (sub, rel, -1), 'label': [obj], 'text_ids': text_ids, 'text_mask': text_mask, 'pred_pos': text_ids.index(103)})
                rel_inv = rel + self.p.num_rel
                text_ids, text_mask = construct_input_text(obj, rel_inv)
                self.triples['train'].append(
                {'triple': (obj, rel_inv, -1), 'label': [sub], 'text_ids': text_ids, 'text_mask': text_mask, 'pred_pos': text_ids.index(103)})

            for split in ['test', 'valid']:
                for sub, rel, obj in tqdm(self.data[split]):
                    text_ids, text_mask = construct_input_text(sub, rel)
                    self.triples['{}_{}'.format(split, 'tail')].append(
                        {'triple': (sub, rel, obj), 'label': self.sr2o_all[(sub, rel)], 'text_ids': text_ids, 'text_mask': text_mask, 'pred_pos': text_ids.index(103)})

                    rel_inv = rel + self.p.num_rel

                    text_ids, text_mask = construct_input_text(obj, rel_inv)
                    self.triples['{}_{}'.format(split, 'head')].append(
                        {'triple': (obj, rel_inv, sub), 'label': self.sr2o_all[(obj, rel_inv)], 'text_ids': text_ids,
                         'text_mask': text_mask, 'pred_pos': text_ids.index(103)})


                print('{}_{} num is {}'.format(split, 'tail', len(self.triples['{}_{}'.format(split, 'tail')])))
                print('{}_{} num is {}'.format(split, 'head', len(self.triples['{}_{}'.format(split, 'head')])))


            self.triples = dict(self.triples)
            json.dump(self.triples, open(triples_save_file, 'w'))

        def get_data_loader(dataset_class, split, batch_size, shuffle=True):
            return DataLoader(
                dataset_class(self.triples[split], self.p),
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=max(0, self.p.num_workers),
                collate_fn=dataset_class.collate_fn
            )

        self.data_iter = {
            'train': get_data_loader(TrainDataset, 'train', self.p.batch_size),
            'valid_head': get_data_loader(TestDataset, 'valid_head', self.p.test_batch_size),
            'valid_tail': get_data_loader(TestDataset, 'valid_tail', self.p.test_batch_size),
            'test_head': get_data_loader(TestDataset, 'test_head', self.p.test_batch_size),
            'test_tail': get_data_loader(TestDataset, 'test_tail', self.p.test_batch_size),
        }
        print('num_ent {} num_rel {}\n'.format(self.p.num_ent, self.p.num_rel))
        print('train set num is {}\n'.format(len(self.triples['train'])))
        print('{}_{} num is {}\n'.format('test', 'tail', len(self.triples['{}_{}'.format('test', 'tail')])))
        print(
            '{}_{} num is {}\n'.format('valid', 'tail', len(self.triples['{}_{}'.format('valid', 'tail')])))


    def __init__(self, params):

        self.p = params
        if self.p.gpu != '-1' and torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.cuda.set_rng_state(torch.cuda.get_rng_state())
            torch.backends.cudnn.deterministic = True
        else:
            self.device = torch.device('cpu')

        self.load_data()

        self.model = MEM_Model(params=self.p, device=self.device)
        self.model.to(self.device)

        if not self.p.fine_tune:
            # don't fine-tune LM
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.p.model_lr, weight_decay=self.p.l2)
        else:
            no_decay = ["bias", "LayerNorm.weight"]
            param_group = [
                {'lr': self.p.model_lr, 'params': [p for n, p in self.model.named_parameters()
                                                if ('plm' not in n) and
                                                (not any(nd in n for nd in no_decay))],
                 'weight_decay': self.p.weight_decay},
                {'lr': self.p.model_lr, 'params': [p for n, p in self.model.named_parameters()
                                                if ('plm' not in n) and
                                                (any(nd in n for nd in no_decay))],
                 'weight_decay': 0.0},
            ]
            param_group += [
                {'lr': self.p.bert_lr, 'params': [p for n, p in self.model.named_parameters()
                                               if ('plm' in n) and
                                               (not any(nd in n for nd in no_decay))],  # name中不包含bias和LayerNorm.weight
                 'weight_decay': self.p.weight_decay},
                {'lr': self.p.bert_lr, 'params': [p for n, p in self.model.named_parameters()
                                               if ('plm' in n) and
                                               (any(nd in n for nd in no_decay))],
                 'weight_decay': 0.0},
            ]

            self.optimizer = AdamW(param_group)  # transformer AdamW


    def read_batch(self, batch, split):
        if split == 'train':
            triple, label, text_ids, text_mask, pred_pos = [_.to(self.device) for _ in batch]
            return triple[:, 0], triple[:, 1], triple[:, 2], label, text_ids, text_mask, pred_pos
        else:
            triple, label, text_ids, text_mask, pred_pos = [_.to(self.device) for _ in batch]
            return triple[:, 0], triple[:, 1], triple[:, 2], label, text_ids, text_mask, pred_pos


    def save_model(self, save_path):

        state = {
            'state_dict': self.model.state_dict(),
            'best_val': self.best_val,
            'best_epoch': self.best_epoch,
            'optimizer': self.optimizer.state_dict(),
            'args': vars(self.p)
        }

        torch.save(state, save_path)



    def load_model(self, load_path):

        state = torch.load(load_path)
        state_dict = state['state_dict']
        self.best_val = state['best_val']
        self.best_val_mrr = self.best_val['mrr']
        self.best_epoch = state['best_epoch']

        self.model.load_state_dict(state_dict)
        self.optimizer.load_state_dict(state['optimizer'])

    def evaluate(self, split, epoch):

        left_lm_results = self.predict(split=split, mode='tail_batch')
        right_lm_results = self.predict(split=split, mode='head_batch')
        log_lm_res, lm_results = get_and_print_combined_results(left_lm_results, right_lm_results)

        # if (epoch + 1) % 1 == 0 or split == 'test':
        print('[Evaluating Epoch {} {}]: \n LM results {}'.format(epoch, split, log_lm_res))

        return lm_results

    def predict(self, split='valid', mode='tail_batch'):

        self.model.eval()

        with torch.no_grad():
            lm_results = {}
            train_iter = iter(self.data_iter['{}_{}'.format(split, mode.split('_')[0])])

            for step, batch in enumerate(train_iter):
                sub, rel, obj, label, text_ids, text_mask, pred_pos = self.read_batch(batch, split)
                output = self.model.forward(sub, rel, text_ids, text_mask, pred_pos)
                b_range = torch.arange(output.size()[0], device=self.device)
                target_output = output[b_range, obj]
                # filter setting
                output = torch.where(label.byte(), -torch.ones_like(output) * 1e9, output)
                output[b_range, obj] = target_output

                lm_ranks = 1 + torch.argsort(torch.argsort(output, dim=1, descending=True), dim=1, descending=False)[
                    b_range, obj]

                lm_ranks = lm_ranks.float()

                lm_results['count'] = torch.numel(lm_ranks) + lm_results.get('count', 0.0)
                lm_results['mr'] = torch.sum(lm_ranks).item() + lm_results.get('mr', 0.0)
                lm_results['mrr'] = torch.sum(1.0 / lm_ranks).item() + lm_results.get('mrr', 0.0)
                for k in range(10):
                    lm_results['hits@{}'.format(k + 1)] = torch.numel(
                        lm_ranks[lm_ranks <= (k + 1)]) + lm_results.get(
                        'hits@{}'.format(k + 1), 0.0)

        return lm_results

    def run_epoch(self, epoch):

        self.model.train()
        losses_train = []
        train_iter = iter(self.data_iter['train'])

        for step, batch in enumerate(train_iter):
            self.optimizer.zero_grad()
            sub, rel, obj, label, text_ids, text_mask, pred_pos = self.read_batch(batch, 'train')

            output = self.model.forward(sub, rel, text_ids, text_mask, pred_pos)

            # Standard CE loss start
            loss = self.model.loss_fn(output, label)
            losses_train.append(loss.item())
            loss.backward()
            # Standard CE loss end

            self.optimizer.step()

            if step % 1000 == 0:
                print(
                    '[E:{}| {}]: Train Loss:{:.5}, Val MRR:{:.5}\t{}'.format(
                        epoch, step,
                        np.mean(losses_train), self.best_val_mrr, self.p.name))

        loss = np.mean(losses_train)

        return loss

    def fit(self):

        try:
            if not self.p.test:
                self.best_val_mrr, self.best_val, self.best_epoch, val_mrr = 0., {}, 0, 0.
                if self.p.load_path != None and self.p.load_epoch > 0:
                    self.save_path = os.path.join('./checkpoints', self.p.load_path)
                    self.load_model(self.save_path)
                    print('Successfully Loaded previous model')
                else:
                    print('Training from Scratch ...')
                    self.save_path = os.path.join('./checkpoints', self.p.name)

                val_results = {}
                val_results['mrr'] = 0
                kill_cnt = 0
                for epoch in range(self.p.load_epoch+1, self.p.max_epochs+1):
                    train_loss = self.run_epoch(epoch)
                    val_results = self.evaluate('valid', epoch)

                    if val_results['mrr'] > self.best_val_mrr:
                        self.best_val = val_results
                        self.best_val_mrr = val_results['mrr']
                        self.best_epoch = epoch
                        self.save_model(self.save_path)
                        kill_cnt = 0
                    else:
                        kill_cnt += 1
                        if kill_cnt > self.p.early_stop:
                            print("Early Stopping!!")
                            break
                    print('[Epoch {}]: Training Loss: {:.5}, Best valid MRR: {:.5}\n\n'.format(
                        epoch, train_loss, self.best_val_mrr))

            else:
                self.save_path = os.path.join('./checkpoints', self.p.load_path)
                print('Loading best model, Evaluating on Test data')
                self.load_model(self.save_path)
                self.evaluate('test', self.best_epoch)
        except Exception as e:
            print ("%s____%s\n"
                              "traceback.format_exc():____%s" % (Exception, e, traceback.format_exc()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser For Arguments',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-name', default='testrun', help='Set run name for saving/restoring models')
    parser.add_argument('-data', dest='dataset', default='FB15k-237', help='Dataset to use, default: FB15k-237')
    # opn is new hyperparameter
    parser.add_argument('-batch', dest='batch_size', default=2048, type=int, help='Batch size')
    parser.add_argument('-test_batch', dest='test_batch_size', default=2048, type=int,
                        help='Batch size of valid and test data')
    parser.add_argument('-epoch', dest='max_epochs', type=int, default=1500, help='Number of epochs')
    parser.add_argument('-l2', type=float, default=0.0, help='L2 Regularization for Optimizer')
    parser.add_argument('--bert_lr', type=float, default=1e-5)
    parser.add_argument('--model_lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-7)

    parser.add_argument('-lbl_smooth', dest='lbl_smooth', type=float, default=0.1, help='Label Smoothing')
    parser.add_argument('-num_workers', type=int, default=10, help='Number of processes to construct batches')
    parser.add_argument('-seed', dest='seed', default=41504, type=int, help='Seed for randomization')



    parser.add_argument('-early_stop', dest="early_stop", default=200, type=int, help="number of early_stop")
    parser.add_argument('-gpu', type=int, default=6, help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')

    ## LLM params
    parser.add_argument('-pretrained_model', type=str, default='bert_base', choices = ['bert_large', 'bert_base'])
    parser.add_argument('-text_len', default=72, type=int, help='')
    parser.add_argument('-desc_max_length', default=40, type=int, help='')
    parser.add_argument('-fine_tune', dest='fine_tune', action='store_true', help='If fine-tune LM or not')
    parser.add_argument('-test', dest='test', action='store_true', help='test the model')
    parser.add_argument('-load_epoch', type=int, default=0)
    parser.add_argument('-load_path', type=str, default=None)
    args = parser.parse_args()

    if args.load_path == None and args.load_epoch == 0:
        args.name = args.name + '_' + time.strftime('%d_%m_%Y') + '_' + time.strftime('%H:%M:%S')

    if args.pretrained_model == 'bert_large':
        args.pretrained_model = '/home/zjlab/gengyx/LMs/BERT_large'
    elif args.pretrained_model == 'bert_base':
        args.pretrained_model = '/home/zjlab/gengyx/LMs/BERT_base'
    args.vocab_size = AutoConfig.from_pretrained(args.pretrained_model).vocab_size
    args.model_dim = AutoConfig.from_pretrained(args.pretrained_model).hidden_size

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.set_device(args.gpu)
        print("GPU is available!")

    model = Runner(args)
    model.fit()
