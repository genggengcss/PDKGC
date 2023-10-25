from helper import *
from data_loader import *
from model import *
import transformers
from transformers import AutoConfig, BertTokenizer, RobertaTokenizer
transformers.logging.set_verbosity_error()
from tqdm import tqdm
import traceback


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
            # rel_text += self.tok.mask_token
            tokenized_text = self.tok(sub_text, text_pair=rel_text, max_length=self.p.text_len-1, truncation=True)
            source_ids = tokenized_text.input_ids  # encoded tokens
            source_mask = tokenized_text.attention_mask  # the position of padding is zero, others are one
            # manually add [MASK] token
            source_ids.insert(-1, self.mask_token_id)
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
        self.p.embed_dim = self.p.k_w * self.p.k_h if self.p.embed_dim is None else self.p.embed_dim
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
        if self.p.pretrained_model_name.lower() == 'bert_base' or self.p.pretrained_model_name.lower() == 'bert_large':
            self.tok = BertTokenizer.from_pretrained(self.p.pretrained_model, add_prefix_space=False)
            triples_save_file_name = 'bert'
        elif self.p.pretrained_model_name.lower() == 'roberta_base' or self.p.pretrained_model_name.lower() == 'roberta_large':
            self.tok = RobertaTokenizer.from_pretrained(self.p.pretrained_model, add_prefix_space=False)
            triples_save_file_name = 'roberta'

        triples_save_file = '../data/{}/{}_{}.txt'.format(self.p.dataset, 'loaded_triples', triples_save_file_name)

        if os.path.exists(triples_save_file):
            self.triples = json.load(open(triples_save_file))
        else:
            self.triples = ddict(list)
            for sub, rel, obj in tqdm(self.data['train']):
                text_ids, text_mask = construct_input_text(sub, rel)
                self.triples['train'].append({'triple': (sub, rel, -1), 'label': [obj], 'sub_samp': 1, 'text_ids': text_ids, 'text_mask': text_mask, 'pred_pos': text_ids.index(self.mask_token_id)})
                rel_inv = rel + self.p.num_rel
                text_ids, text_mask = construct_input_text(obj, rel_inv)
                self.triples['train'].append(
                {'triple': (obj, rel_inv, -1), 'label': [sub], 'sub_samp': 1, 'text_ids': text_ids, 'text_mask': text_mask, 'pred_pos': text_ids.index(self.mask_token_id)})

            for split in ['test', 'valid']:
                for sub, rel, obj in tqdm(self.data[split]):
                    text_ids, text_mask = construct_input_text(sub, rel)
                    self.triples['{}_{}'.format(split, 'tail')].append(
                        {'triple': (sub, rel, obj), 'label': self.sr2o_all[(sub, rel)], 'text_ids': text_ids, 'text_mask': text_mask, 'pred_pos': text_ids.index(self.mask_token_id)})


                    rel_inv = rel + self.p.num_rel

                    text_ids, text_mask = construct_input_text(obj, rel_inv)
                    self.triples['{}_{}'.format(split, 'head')].append(
                        {'triple': (obj, rel_inv, sub), 'label': self.sr2o_all[(obj, rel_inv)], 'text_ids': text_ids,
                         'text_mask': text_mask, 'pred_pos': text_ids.index(self.mask_token_id)})


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
        self.edge_index, self.edge_type = self.construct_adj()

    def construct_adj(self):

        edge_index, edge_type = [], []

        for sub, rel, obj in self.data['train']:
            edge_index.append((sub, obj))
            edge_type.append(rel)

        # Adding inverse edges
        for sub, rel, obj in self.data['train']:
            edge_index.append((obj, sub))
            edge_type.append(rel + self.p.num_rel)
        # edge_index: 2 * 2E, edge_type: 2E * 1
        edge_index = torch.LongTensor(edge_index).to(self.device).t()
        edge_type = torch.LongTensor(edge_type).to(self.device)

        return edge_index, edge_type

    def __init__(self, params):

        self.p = params

        if self.p.gpu != '-1' and torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.cuda.set_rng_state(torch.cuda.get_rng_state())
            torch.backends.cudnn.deterministic = True
        else:
            self.device = torch.device('cpu')

        # [MASK] token id is 103 in bert and <mask> token id is 50264 in roberta
        if self.p.pretrained_model_name.lower() == 'bert_base' or self.p.pretrained_model_name.lower() == 'bert_large':
            self.mask_token_id = 103
        elif self.p.pretrained_model_name.lower() == 'roberta_base' or self.p.pretrained_model_name.lower() == 'roberta_large':
            self.mask_token_id = 50264
        self.load_data()
        self.model = self.add_model(self.p.model, self.p.score_func)
        self.optimizer, self.optimizer_mi = self.add_optimizer(self.model)

        self.best_val_mrr = {'combine': 0., 'struc': 0., 'text': 0.}
        self.best_epoch = {'combine': 0, 'struc': 0, 'text': 0}

        os.makedirs('./checkpoints', exist_ok=True)

        if self.p.load_path != None and self.p.load_epoch > 0 and self.p.load_type != '':
            self.path_template = os.path.join('./checkpoints', self.p.load_path)
            path = self.path_template + '_type_{0}_epoch_{1}'.format(self.p.load_type, self.p.load_epoch)
            self.load_model(path)
            print('Successfully Loaded previous model')
        else:
            print('Training from Scratch ...')
            self.path_template = os.path.join('./checkpoints', self.p.name)


    def add_model(self, model, score_func):
        """
        Creates the computational graph

        Parameters
        ----------
        model_name:     Contains the model name to be created

        Returns
        -------
        Creates the computational graph for model and initializes it

        """

        model_name = '{}_{}'.format(model, score_func)

        if model_name.lower() == 'disenkgat_transe':
            model = DisenKGAT_TransE(self.edge_index, self.edge_type, params=self.p)
        elif model_name.lower() == 'disenkgat_distmult':
            model = DisenKGAT_DistMult(self.edge_index, self.edge_type, params=self.p)
        elif model_name.lower() == 'disenkgat_conve':
            model = DisenKGAT_ConvE(self.edge_index, self.edge_type, params=self.p)

            # if we want to load pretrained DisenKGAT_ConvE
        elif model_name.lower() == 'pretrained_disenkgat_conve':
            if self.p.dataset =='FB15k-237':
                model_save_path = '/home/zjlab/gengyx/KGE/DisenKGAT-2023/checkpoints/ConvE_FB15k_K4_D200_club_b_mi_drop_200d_08_09_2023_19:21:24'
            elif self.p.dataset == 'WN18RR':
                model_save_path = '/home/zjlab/gengyx/KGE/DisenKGAT-2023/checkpoints/ConvE_wn18rr_K2_D200_club_b_mi_drop_200d_27_09_2023_17:12:54'
            state = torch.load(model_save_path, map_location=self.device)
            pretrained_dict = state['state_dict']
            model = DisenKGAT_ConvE(self.edge_index, self.edge_type, params=self.p)
            model_dict = model.state_dict()
            pretrained_dict = {k:v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
        else:
            raise NotImplementedError

        model.to(self.device)
        return model

    def add_optimizer(self, model):


        mi_disc_params = list(map(id, model.mi_Discs.parameters()))
        rest_params = filter(lambda x: id(x) not in mi_disc_params, model.parameters())
        # for m in model.mi_Discs.modules():
        #     print(m)
        # for name, parameters in model.named_parameters():
        #     print(name, ':', parameters.size(), parameters.requires_grad)
        return torch.optim.Adam(rest_params, lr=self.p.lr, weight_decay=self.p.l2), torch.optim.Adam(
            model.mi_Discs.parameters(), lr=self.p.lr, weight_decay=self.p.l2)



    def read_batch(self, batch, split):
        if split == 'train':
            triple, label, text_ids, text_mask, pred_pos = [_.to(self.device) for _ in batch]
            return triple[:, 0], triple[:, 1], triple[:, 2], label, text_ids, text_mask, pred_pos
        else:
            triple, label, text_ids, text_mask, pred_pos = [_.to(self.device) for _ in batch]
            return triple[:, 0], triple[:, 1], triple[:, 2], label, text_ids, text_mask, pred_pos


    # def load_model(self, load_path):
    #
    #     state = torch.load(load_path)
    #     state_dict = state['state_dict']
    #     # self.best_val = state['best_val']
    #     # self.best_val_mrr = self.best_val['mrr']
    #     # self.best_val_mrr['struc'] =
    #     self.model.load_state_dict(state_dict)
    #     self.optimizer.load_state_dict(state['optimizer'])

    def load_model(self, load_path):

        state = torch.load(load_path, map_location=self.device)
        state_dict = state['state_dict']
        self.best_val_mrr[self.p.load_type] = state['best_val_mrr']
        self.best_epoch[self.p.load_type] = self.p.load_epoch
        self.model.load_state_dict(state_dict)
        self.optimizer.load_state_dict(state['optimizer'])

    def evaluate(self, split, epoch):

        left_combine_results, left_struc_results, left_lm_results = self.predict(split=split, mode='tail_batch')
        right_combine_results, right_struc_results, right_lm_results = self.predict(split=split, mode='head_batch')
        log_combine_res, combine_results = get_and_print_combined_results(left_combine_results, right_combine_results)
        log_struc_res, struc_results = get_and_print_combined_results(left_struc_results, right_struc_results)
        log_lm_res, lm_results = get_and_print_combined_results(left_lm_results, right_lm_results)

        # if (epoch + 1) % 1 == 0 or split == 'test':
        print('[Evaluating Epoch {} {}]: \n COMBINE results {}'.format(epoch, split, log_combine_res))

        print('Structure Results {}'.format(log_struc_res))
        print('LM Results {}'.format(log_lm_res))


        return combine_results, struc_results, lm_results

    def predict(self, split='valid', mode='tail_batch'):

        self.model.eval()

        with torch.no_grad():
            combine_results = {}
            lm_results = {}
            struc_results = {}
            train_iter = iter(self.data_iter['{}_{}'.format(split, mode.split('_')[0])])

            for step, batch in enumerate(train_iter):
                sub, rel, obj, label, text_ids, text_mask, pred_pos = self.read_batch(batch, split)
                pred, output, _ = self.model.forward(sub, rel, text_ids, text_mask, pred_pos, split)
                b_range = torch.arange(pred.size()[0], device=self.device)
                target_output = output[b_range, obj]
                target_pred = pred[b_range, obj]
                # filter setting
                pred = torch.where(label.byte(), -torch.ones_like(pred) * 1e9, pred) # 满足条件返回a，不满足返回b
                output = torch.where(label.byte(), -torch.ones_like(output) * 1e9, output) # 满足条件返回a，不满足返回b
                output[b_range, obj] = target_output
                pred[b_range, obj] = target_pred

                struc_ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[
                    b_range, obj]
                lm_ranks = 1 + torch.argsort(torch.argsort(output, dim=1, descending=True), dim=1, descending=False)[
                    b_range, obj]


                struc_ranks1 = torch.unsqueeze(struc_ranks, 1)
                lm_ranks1 = torch.unsqueeze(lm_ranks, 1)
                combine_ranks = torch.cat((struc_ranks1, lm_ranks1), 1)
                combine_ranks = torch.squeeze(torch.min(combine_ranks, 1)[0])


                combine_ranks = combine_ranks.float()
                struc_ranks = struc_ranks.float()
                lm_ranks = lm_ranks.float()


                combine_results['count'] = torch.numel(combine_ranks) + combine_results.get('count', 0.0)
                combine_results['mr'] = torch.sum(combine_ranks).item() + combine_results.get('mr', 0.0)
                combine_results['mrr'] = torch.sum(1.0 / combine_ranks).item() + combine_results.get('mrr', 0.0)
                for k in range(10):
                    combine_results['hits@{}'.format(k + 1)] = torch.numel(combine_ranks[combine_ranks <= (k + 1)]) + combine_results.get(
                        'hits@{}'.format(k + 1), 0.0)

                struc_results['count'] = torch.numel(struc_ranks) + struc_results.get('count', 0.0)
                struc_results['mr'] = torch.sum(struc_ranks).item() + struc_results.get('mr', 0.0)
                struc_results['mrr'] = torch.sum(1.0 / struc_ranks).item() + struc_results.get('mrr', 0.0)
                for k in range(10):
                    struc_results['hits@{}'.format(k + 1)] = torch.numel(
                        struc_ranks[struc_ranks <= (k + 1)]) + struc_results.get(
                        'hits@{}'.format(k + 1), 0.0)

                lm_results['count'] = torch.numel(lm_ranks) + lm_results.get('count', 0.0)
                lm_results['mr'] = torch.sum(lm_ranks).item() + lm_results.get('mr', 0.0)
                lm_results['mrr'] = torch.sum(1.0 / lm_ranks).item() + lm_results.get('mrr', 0.0)
                for k in range(10):
                    lm_results['hits@{}'.format(k + 1)] = torch.numel(
                        lm_ranks[lm_ranks <= (k + 1)]) + lm_results.get(
                        'hits@{}'.format(k + 1), 0.0)


        return combine_results, struc_results, lm_results

    def run_epoch(self, epoch):

        self.model.train()
        losses = []
        losses_struc = []
        losses_lm = []
        corr_losses = []

        lld_losses = []
        train_iter = iter(self.data_iter['train'])

        for step, batch in enumerate(train_iter):
            self.optimizer.zero_grad()
            # if self.p.mi_train and self.p.mi_method.startswith('club'):
            self.model.mi_Discs.eval()
            sub, rel, obj, label, text_ids, text_mask, pred_pos = self.read_batch(batch, 'train')

            pred, output, corr = self.model.forward(sub, rel, text_ids, text_mask, pred_pos, 'train')

            loss_struc = self.model.loss_fn(pred, label)
            loss_lm = self.model.loss_fn(output, label)
            # if self.p.mi_train:
            losses_struc.append(loss_struc.item())
            losses_lm.append(loss_lm.item())

            # weighted sum:
            if self.p.loss_weight:
                loss_weighted_sum = self.model.loss_weight(loss_struc, loss_lm)
                loss = loss_weighted_sum + self.p.alpha * corr
            # simple sum
            else:
                loss = loss_struc + loss_lm + self.p.alpha * corr

            corr_losses.append(corr.item())

            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            # start to compute mi_loss
            # if self.p.mi_train and self.p.mi_method.startswith('club'):
            for i in range(self.p.mi_epoch):
                self.model.mi_Discs.train()
                lld_loss = self.model.lld_best(sub, rel)
                self.optimizer_mi.zero_grad()
                lld_loss.backward()
                self.optimizer_mi.step()
                lld_losses.append(lld_loss.item())

            if step % 1000 == 0:
                # if self.p.mi_train:
                print(
                    '[E:{}| {}]: Total Loss:{:.5}, Train {} Loss:{:.5}, Train MASK Loss:{:.5}, Corr Loss:{:.5}\t{} \n \t Best Combine Valid MRR: {:.5} \t Best Struc Valid MRR: {:.5} \t Best LM Valid MRR: {:.5}'.format(
                        epoch, step, np.mean(losses), self.p.score_func, np.mean(losses_struc), np.mean(losses_lm), np.mean(corr_losses), self.p.name, self.best_val_mrr['combine'], self.best_val_mrr['struc'], self.best_val_mrr['text']))
        loss = np.mean(losses_struc)
        # if self.p.mi_train:
        loss_corr = np.mean(corr_losses)
        if self.p.mi_method.startswith('club') and self.p.mi_epoch == 1:
            loss_lld = np.mean(lld_losses)
            return loss, loss_corr, loss_lld
        return loss, loss_corr, 0.

    def save_model(self, results, type, epoch):
        if results['mrr'] > self.best_val_mrr[type]:
            last_path = self.path_template + '_type_{0}_epoch_{1}'.format(type, self.best_epoch[type])
            save_path = self.path_template + '_type_{0}_epoch_{1}'.format(type, epoch)

            self.best_epoch[type] = epoch
            self.best_val_mrr[type] = results['mrr']

            if os.path.exists(last_path) and last_path != save_path:
                os.remove(last_path)
                print('Last Saved Model {} deleted'.format(last_path))
            torch.save({
                'state_dict': self.model.state_dict(),
                'best_val_mrr': self.best_val_mrr[type],
                # 'best_epoch': self.best_epoch,
                'optimizer': self.optimizer.state_dict(),
                # 'args': vars(self.p)
            }, save_path)

    def fit(self):

        try:
            if not self.p.test:

                kill_cnt = 0
                for epoch in range(self.p.load_epoch+1, self.p.max_epochs+1):
                    train_loss, corr_loss, lld_loss = self.run_epoch(epoch)
                    # if ((epoch + 1) % 10 == 0):
                    combine_val_results, struc_val_results, lm_val_results = self.evaluate('valid', epoch)
                    if combine_val_results['mrr'] <= self.best_val_mrr['combine']:
                        kill_cnt += 1
                        if kill_cnt % 10 == 0 and self.p.gamma > self.p.max_gamma:
                            self.p.gamma -= 5
                            print('Gamma decay on saturation, updated value of gamma: {}'.format(self.p.gamma))
                        if kill_cnt > self.p.early_stop:
                            print("Early Stopping!!")
                            break
                    else:
                        kill_cnt = 0
                    self.save_model(combine_val_results, 'combine', epoch)
                    self.save_model(struc_val_results, 'struc', epoch)
                    self.save_model(lm_val_results, 'text', epoch)
                    # if self.p.mi_train:
                    if self.p.mi_method == 'club_s' or self.p.mi_method == 'club_b':
                        print(
                            '[Epoch {}]: Training Loss: {:.5}, corr Loss: {:.5}, lld loss :{:.5}, \n \t Best Combine Valid MRR: {:.5} \n \t Best Struc Valid MRR: {:.5} \n \tBest LM Valid MRR: {:.5}\n\n'.format(
                                epoch, train_loss, corr_loss,
                                lld_loss, self.best_val_mrr['combine'], self.best_val_mrr['struc'], self.best_val_mrr['text']))
                    else:
                        print(
                            '[Epoch {}]: Training Loss: {:.5}, corr Loss: {:.5}, \n\t Best Combine Valid MRR: {:.5} \n \tBest Struc Valid MRR: {:.5} \n \tBest LM Valid MRR: {:.5}\n\n'.format(
                                epoch, train_loss, corr_loss,
                                self.best_val_mrr['combine'], self.best_val_mrr['struc'], self.best_val_mrr['text']))

            else:
                # self.path_template = os.path.join('./checkpoints', self.p.load_path)
                # path = self.path_template + '_type_{0}_epoch_{1}'.format(self.p.load_type, self.p.load_epoch)
                print('Loading best model, Evaluating on Test data')
                # self.load_model(path)
                self.evaluate('test', self.best_epoch[self.p.load_type])
        except Exception as e:
            print ("%s____%s\n"
                              "traceback.format_exc():____%s" % (Exception, e, traceback.format_exc()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser For Arguments',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-name', default='testrun', help='Set run name for saving/restoring models')
    parser.add_argument('-data', dest='dataset', default='FB15k-237', help='Dataset to use, default: FB15k-237')
    parser.add_argument('-model', dest='model', default='disenkgat', help='Model Name')
    parser.add_argument('-score_func', dest='score_func', default='conve', help='Score Function for Link prediction')
    parser.add_argument('-opn', dest='opn', default='cross', help='Composition Operation to be used in RAGAT')
    parser.add_argument('-loss_weight', dest='loss_weight', action='store_true', help='whether to use weighted loss')
    # opn is new hyperparameter
    parser.add_argument('-batch', dest='batch_size', default=2048, type=int, help='Batch size')
    parser.add_argument('-test_batch', dest='test_batch_size', default=2048, type=int,
                        help='Batch size of valid and test data')
    parser.add_argument('-gamma', type=float, default=9.0, help='Margin')
    parser.add_argument('-epoch', dest='max_epochs', type=int, default=1500, help='Number of epochs')
    parser.add_argument('-l2', type=float, default=0.0, help='L2 Regularization for Optimizer')
    parser.add_argument('-lr', type=float, default=0.001, help='Starting Learning Rate')
    parser.add_argument('-lbl_smooth', dest='lbl_smooth', type=float, default=0.1, help='Label Smoothing')
    parser.add_argument('-num_workers', type=int, default=10, help='Number of processes to construct batches')
    parser.add_argument('-seed', dest='seed', default=41504, type=int, help='Seed for randomization')

    parser.add_argument('-bias', dest='bias', action='store_true', help='Whether to use bias in the model')

    parser.add_argument('-embed_dim', dest='embed_dim', default=156, type=int,
                        help='Embedding dimension to give as input to score function')
    parser.add_argument('-gcn_layer', dest='gcn_layer', default=1, type=int, help='Number of GCN Layers to use')
    parser.add_argument('-gcn_drop', dest='dropout', default=0.4, type=float, help='Dropout to use in GCN Layer')
    parser.add_argument('-hid_drop', dest='hid_drop', default=0.3, type=float, help='Dropout after GCN')

    # ConvE specific hyperparameters
    parser.add_argument('-hid_drop2', dest='hid_drop2', default=0.3, type=float, help='ConvE: Hidden dropout')
    parser.add_argument('-feat_drop', dest='feat_drop', default=0.3, type=float, help='ConvE: Feature Dropout')
    parser.add_argument('-k_w', dest='k_w', default=12, type=int, help='ConvE: k_w')
    parser.add_argument('-k_h', dest='k_h', default=13, type=int, help='ConvE: k_h')
    parser.add_argument('-num_filt', dest='num_filt', default=200, type=int,
                        help='ConvE: Number of filters in convolution')
    parser.add_argument('-ker_sz', dest='ker_sz', default=7, type=int, help='ConvE: Kernel size to use')

    parser.add_argument('-logdir', dest='log_dir', default='./log/', help='Log directory')
    parser.add_argument('-config', dest='config_dir', default='./config/', help='Config directory')
    parser.add_argument('-num_bases', dest='num_bases', default=-1, type=int,
                        help='Number of basis relation vectors to use')
    parser.add_argument('-head_num', dest="head_num", default=1, type=int, help="Number of attention heads")
    parser.add_argument('-num_factors', dest="num_factors", default=4, type=int, help="Number of factors")
    parser.add_argument('-alpha', dest="alpha", default=1e-1, type=float, help='Dropout for Feature')
    parser.add_argument('-early_stop', dest="early_stop", default=200, type=int, help="number of early_stop")
    parser.add_argument('-no_act', dest='no_act', action='store_true', help='whether to use non_linear function')
    parser.add_argument('-mi_method', dest='mi_method', default='club_b',
                        help='Composition Operation to be used in RAGAT')
    parser.add_argument('-att_mode', dest='att_mode', default='dot_weight',
                        help='Composition Operation to be used in RAGAT')
    parser.add_argument('-mi_epoch', dest="mi_epoch", default=1, type=int, help="Number of MI_Disc training times")
    parser.add_argument('-score_order', dest='score_order', default='after',
                        help='Composition Operation to be used in RAGAT')
    parser.add_argument('-mi_drop', dest='mi_drop', action='store_true', help='whether to use non_linear function')
    parser.add_argument('-max_gamma', type=float, default=5.0, help='Margin')
    parser.add_argument('-fix_gamma', dest='fix_gamma', action='store_true', help='whether to use non_linear function')
    parser.add_argument('-init_gamma', type=float, default=9.0, help='Margin')
    parser.add_argument('-gamma_method', dest='gamma_method', default='norm',
                        help='Composition Operation to be used in RAGAT')
    parser.add_argument('-gpu', type=int, default=6, help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')

    ## LLM params
    parser.add_argument('-pretrained_model', type=str, default='bert_large', choices = ['bert_large', 'bert_base', 'roberta_large', 'roberta_base'])
    parser.add_argument('-pretrained_model_name', type=str, default='bert_large', help='')
    parser.add_argument('-prompt_hidden_dim', default=-1, type=int, help='')
    parser.add_argument('-text_len', default=72, type=int, help='')
    parser.add_argument('-prompt_length', default=10, type=int, help='')
    parser.add_argument('-desc_max_length', default=40, type=int, help='')
    parser.add_argument('-load_epoch', type=int, default=0)
    parser.add_argument('-load_type', type=str, default='')
    parser.add_argument('-load_path', type=str, default=None)
    parser.add_argument('-test', dest='test', action='store_true', help='test the model')
    args = parser.parse_args()

    if args.load_path == None and args.load_epoch == 0 and args.load_type == '':
        args.name = args.name + '_' + time.strftime('%d_%m_%Y') + '_' + time.strftime('%H:%M:%S')

    args.pretrained_model_name = args.pretrained_model    
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
    if args.prompt_hidden_dim == -1:
        args.prompt_hidden_dim = args.embed_dim // 2

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.set_device(args.gpu)
        print("GPU is available!")

    model = Runner(args)
    model.fit()
