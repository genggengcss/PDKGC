from helper import *
import torch.nn as nn
from DisenLayer import *
from transformers import AutoConfig
from bert_for_layerwise import BertModelForLayerwise


class CLUBSample(nn.Module):  # Sampled version of the CLUB estimator
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUBSample, self).__init__()
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size // 2, y_dim))

        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size // 2, y_dim),
                                      nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def loglikeli(self, x_samples, y_samples):
        #         print(x_samples.size())
        #         print(y_samples.size())
        mu, logvar = self.get_mu_logvar(x_samples)

        return (-(mu - y_samples) ** 2 / 2. / logvar.exp()).sum(dim=1).mean(dim=0)

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)

        sample_size = x_samples.shape[0]
        # random_index = torch.randint(sample_size, (sample_size,)).long()
        random_index = torch.randperm(sample_size).long()

        positive = - (mu - y_samples) ** 2 / logvar.exp()
        negative = - (mu - y_samples[random_index]) ** 2 / logvar.exp()
        upper_bound = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
        return upper_bound / 2.

    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)


class BaseModel(torch.nn.Module):
    def __init__(self, params):
        super(BaseModel, self).__init__()

        self.p = params
        self.act = torch.tanh
        self.bceloss = torch.nn.BCELoss()

    def loss(self, pred, true_label):
        return self.bceloss(pred, true_label)


class SparseInputLinear(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(SparseInputLinear, self).__init__()
        weight = np.zeros((inp_dim, out_dim), dtype=np.float32)
        weight = nn.Parameter(torch.from_numpy(weight))
        bias = np.zeros(out_dim, dtype=np.float32)
        bias = nn.Parameter(torch.from_numpy(bias))
        self.inp_dim, self.out_dim = inp_dim, out_dim
        self.weight, self.bias = weight, bias
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):  # *nn.Linear* does not accept sparse *x*.
        return torch.mm(x, self.weight) + self.bias


class CapsuleBase(BaseModel):
    def __init__(self, edge_index, edge_type, num_rel, params=None):
        super(CapsuleBase, self).__init__(params)
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.device = self.edge_index.device
        self.init_embed = get_param((self.p.num_ent, self.p.embed_dim))
        self.init_rel = get_param((num_rel * 2, self.p.embed_dim))
        self.pca = SparseInputLinear(self.p.embed_dim, self.p.num_factors * self.p.embed_dim)
        conv_ls = []
        for i in range(self.p.gcn_layer):
            conv = DisenLayer(self.edge_index, self.edge_type, self.p.embed_dim, self.p.embed_dim, num_rel,
                              act=self.act, params=self.p, head_num=self.p.head_num)
            self.add_module('conv_{}'.format(i), conv)
            conv_ls.append(conv)

        self.conv_ls = conv_ls
        # if self.p.mi_train:
        if self.p.mi_method == 'club_b':
            num_dis = int((self.p.num_factors) * (self.p.num_factors - 1) / 2)
            # print(num_dis)
            self.mi_Discs = nn.ModuleList(
                [CLUBSample(self.p.embed_dim, self.p.embed_dim, self.p.embed_dim) for fac in range(num_dis)])
        elif self.p.mi_method == 'club_s':
            self.mi_Discs = nn.ModuleList(
                [CLUBSample((fac + 1) * self.p.embed_dim, self.p.embed_dim, (fac + 1) * self.p.embed_dim) for fac in
                 range(self.p.num_factors - 1)])

        self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))
        # self.rel_drop = nn.Dropout(0.1)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def lld_bst(self, sub, rel, drop1, mode='train'):
        x = self.act(self.pca(self.init_embed)).view(-1, self.p.num_factors, self.p.embed_dim)  # [N K F]
        r = self.init_rel
        for conv in self.conv_ls:
            x, r = conv(x, r, mode)  # N K F
            if self.p.mi_drop:
                x = drop1(x)
            else:
                continue

        sub_emb = torch.index_select(x, 0, sub)
        lld_loss = 0.
        sub_emb = sub_emb.view(-1, self.p.embed_dim * self.p.num_factors)
        if self.p.mi_method == 'club_s':
            for i in range(self.p.num_factors - 1):
                bnd = i + 1
                lld_loss += self.mi_Discs[i].learning_loss(sub_emb[:, :bnd * self.p.embed_dim], sub_emb[:,
                                                                                                bnd * self.p.embed_dim: (
                                                                                                                                    bnd + 1) * self.p.embed_dim])

        elif self.p.mi_method == 'club_b':
            cnt = 0
            for i in range(self.p.num_factors):
                for j in range(i + 1, self.p.num_factors):
                    lld_loss += self.mi_Discs[cnt].learning_loss(
                        sub_emb[:, i * self.p.embed_dim: (i + 1) * self.p.embed_dim],
                        sub_emb[:, j * self.p.embed_dim: (j + 1) * self.p.embed_dim])
                    cnt += 1
        return lld_loss

    def mi_cal(self, sub_emb):

        def loss_dependence_club_s(sub_emb):
            mi_loss = 0.
            for i in range(self.p.num_factors - 1):
                bnd = i + 1
                mi_loss += self.mi_Discs[i](sub_emb[:, :bnd * self.p.embed_dim],
                                            sub_emb[:, bnd * self.p.embed_dim: (bnd + 1) * self.p.embed_dim])
            return mi_loss

        def loss_dependence_club_b(sub_emb):
            mi_loss = 0.
            cnt = 0
            for i in range(self.p.num_factors):
                for j in range(i + 1, self.p.num_factors):
                    mi_loss += self.mi_Discs[cnt](sub_emb[:, i * self.p.embed_dim: (i + 1) * self.p.embed_dim],
                                                  sub_emb[:, j * self.p.embed_dim: (j + 1) * self.p.embed_dim])
                    cnt += 1
            return mi_loss

        if self.p.mi_method == 'club_s':
            mi_loss = loss_dependence_club_s(sub_emb)
        elif self.p.mi_method == 'club_b':
            mi_loss = loss_dependence_club_b(sub_emb)
        else:
            raise NotImplementedError

        return mi_loss

    def forward_base(self, sub, rel, drop1, mode):
        # if not self.p.no_enc:
        x = self.act(self.pca(self.init_embed)).view(-1, self.p.num_factors, self.p.embed_dim)  # [N K F]
        r = self.init_rel
        for conv in self.conv_ls:
            x, r = conv(x, r, mode)  # N K F
            x = drop1(x)
        sub_emb = torch.index_select(x, 0, sub)
        rel_emb = torch.index_select(self.init_rel, 0, rel).repeat(1, self.p.num_factors)
        rel_emb_single = torch.index_select(self.init_rel, 0, rel)
        sub_emb = sub_emb.view(-1, self.p.embed_dim * self.p.num_factors)
        mi_loss = self.mi_cal(sub_emb)

        return sub_emb, rel_emb, x, mi_loss, rel_emb_single

    def test_base(self, sub, rel, drop1, mode):
        x = self.act(self.pca(self.init_embed)).view(-1, self.p.num_factors, self.p.embed_dim)  # [N K F]
        r = self.init_rel
        for conv in self.conv_ls:
            x, r = conv(x, r, mode)  # N K F
            x = drop1(x)

        sub_emb = torch.index_select(x, 0, sub)
        rel_emb = torch.index_select(self.init_rel, 0, rel).repeat(1, self.p.num_factors)
        rel_emb_single = torch.index_select(self.init_rel, 0, rel)

        return sub_emb, rel_emb, x, 0., rel_emb_single


class Prompter(nn.Module):
    def __init__(self, plm_config, embed_dim, prompt_length):
        super().__init__()
        self.embed_dim = embed_dim
        self.model_dim = plm_config.hidden_size
        self.num_heads = plm_config.num_attention_heads
        self.num_layers = plm_config.num_hidden_layers
        self.prompt_length = prompt_length
        self.prompt_hidden_dim = plm_config.prompt_hidden_dim
        self.dropout = nn.Dropout(plm_config.hidden_dropout_prob)
        self.seq = nn.Sequential(
            nn.Linear(embed_dim, self.prompt_hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(self.prompt_hidden_dim, self.model_dim * self.num_layers * prompt_length, bias=False),
        )

    def forward(self, x):
        # out -- .shape: (batch_size, 2, model_dim * num_layers * prompt_length * 2)
        out = self.seq(x)
        # out-- reshape: (batch_size, num_layers, prompt_length * 2, model_dim)
        out = out.view(x.size(0), self.num_layers, -1, self.model_dim)
        out = self.dropout(out)
        return out


class DisenKGAT_ConvE(CapsuleBase):
    def __init__(self, edge_index, edge_type, params=None):
        super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)

        self.embed_dim = self.p.embed_dim

        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(self.p.num_filt)
        self.bn2 = torch.nn.BatchNorm1d(self.embed_dim)

        self.hidden_drop = torch.nn.Dropout(self.p.hid_drop)
        self.hidden_drop2 = torch.nn.Dropout(self.p.hid_drop2)
        self.feature_drop = torch.nn.Dropout(self.p.feat_drop)
        self.m_conv1 = torch.nn.Conv2d(1, out_channels=self.p.num_filt, kernel_size=(self.p.ker_sz, self.p.ker_sz),
                                       stride=1, padding=0, bias=self.p.bias)

        flat_sz_h = int(2 * self.p.k_w) - self.p.ker_sz + 1
        flat_sz_w = self.p.k_h - self.p.ker_sz + 1
        self.flat_sz = flat_sz_h * flat_sz_w * self.p.num_filt
        self.fc = torch.nn.Linear(self.flat_sz, self.embed_dim)

        self.rel_weight = self.conv_ls[-1].rel_weight

        self.plm_configs = AutoConfig.from_pretrained(self.p.pretrained_model)
        self.plm_configs.prompt_length = self.p.prompt_length
        self.plm_configs.prompt_hidden_dim = self.p.prompt_hidden_dim
        self.plm = BertModelForLayerwise.from_pretrained(self.p.pretrained_model)
        self.prompter = Prompter(self.plm_configs, self.p.embed_dim, self.p.prompt_length)
        self.llm_fc = nn.Linear(self.p.prompt_length * self.plm_configs.hidden_size, self.p.embed_dim)
        if self.p.prompt_length > 0:
            for p in self.plm.parameters():
                p.requires_grad = False

        self.loss_fn = get_loss_fn(params)

    def concat(self, e1_embed, rel_embed):
        e1_embed = e1_embed.view(-1, 1, self.embed_dim)
        rel_embed = rel_embed.view(-1, 1, self.embed_dim)
        stack_inp = torch.cat([e1_embed, rel_embed], 1)
        stack_inp = torch.transpose(stack_inp, 2, 1).reshape((-1, 1, 2 * self.p.k_w, self.p.k_h))
        return stack_inp

    def lld_best(self, sub, rel):
        return self.lld_bst(sub, rel, self.hidden_drop)

    def forward(self, sub, rel, text_ids, text_mask, mode='train'):
        if mode == 'train':
            sub_emb, rel_emb, all_ent, corr, rel_emb_single = self.forward_base(sub, rel, self.hidden_drop, mode)
            # sub_emb = sub_emb.view(-1, self.p.num_factors, self.p.embed_dim)
        else:
            sub_emb, rel_emb, all_ent, corr, rel_emb_single = self.test_base(sub, rel, self.hidden_drop, mode)

        # # method(1)(2), start
        # text_ids = text_ids.repeat(1, self.p.num_factors)
        # text_mask = text_mask.repeat(1, self.p.num_factors)
        # text_ids = text_ids.view(-1, self.p.text_len)
        # text_mask = text_mask.view(-1, self.p.text_len)
        #
        all_ent = all_ent.view(-1, self.p.num_factors, self.p.embed_dim)

        sub_emb = sub_emb.view(-1, self.p.num_factors, self.p.embed_dim)
        rel_emb_exd = torch.unsqueeze(rel_emb_single, 1)
        embed_input = torch.cat([sub_emb, rel_emb_exd], dim=1)

        # ## plug LLMs
        prompt = self.prompter(embed_input)
        prompt_attention_mask = torch.ones(sub_emb.size(0), self.p.prompt_length * (self.p.num_factors + 1)).type_as(text_mask)
        text_mask = torch.cat((prompt_attention_mask, text_mask), dim=1)
        output = self.plm(input_ids=text_ids, attention_mask=text_mask, layerwise_prompt=prompt)
        # # last_hidden_state -- .shape: (batch_size, seq_len, model_dim)
        last_hidden_state = output.last_hidden_state
        ent_rel_state = last_hidden_state[:, :self.p.prompt_length * (self.p.num_factors + 1)]
        plm_embeds = torch.chunk(ent_rel_state, chunks=(self.p.num_factors + 1), dim=1)
        plm_sub_embeds, plm_rel_embed = plm_embeds[:self.p.num_factors], plm_embeds[-1]

        plm_sub_embed = torch.stack(plm_sub_embeds, dim=1)
        plm_sub_embed = self.llm_fc(plm_sub_embed.reshape(sub_emb.size(0), self.p.num_factors, -1))
        plm_rel_embed = self.llm_fc(plm_rel_embed.reshape(rel_emb.size(0), -1)).repeat(1, self.p.num_factors)

        # for i in range(self.p.num_factors):
        #     plm_sub_embed_i = self.llm_fc(plm_sub_embeds[i].reshape(rel_emb.size(0), -1))

        # print(plm_sub_embed.shape)
        # print(plm_rel_embed.shape)
        plm_rel_embed = plm_rel_embed.view(-1, self.p.num_factors, self.p.embed_dim)
        attention = self.leakyrelu(torch.einsum('bkf,bkf->bk', [plm_sub_embed, plm_rel_embed]))
        attention = nn.Softmax(dim=-1)(attention)

        plm_sub_embed = plm_sub_embed.view(-1, self.p.embed_dim)
        plm_rel_embed = plm_rel_embed.view(-1, self.p.embed_dim)

        stk_inp = self.concat(plm_sub_embed, plm_rel_embed)
        x = self.bn0(stk_inp)
        x = self.m_conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_drop(x)
        x = x.view(-1, self.flat_sz)
        x = self.fc(x)
        x = self.hidden_drop2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = x.view(-1, self.p.num_factors, self.p.embed_dim)

        x = torch.einsum('bkf,nkf->bkn', [x, all_ent])
        x += self.bias.expand_as(x)

        # before
        logist = torch.einsum('bk,bkn->bn', [attention, x])
        # pred = torch.sigmoid(x)
        #
        # if self.p.score_order == 'before':
        #     x = torch.einsum('bk,bkn->bn', [attention, x])
        #     pred = torch.sigmoid(x)
        # elif self.p.score_order == 'after':
        #     x = torch.sigmoid(x)
        #     pred = torch.einsum('bk,bkn->bn', [attention, x])
        #     pred = torch.clamp(pred, min=0., max=1.0)
        #
        # # method(1)(2), end
        # start to calculate the attention

        return logist, corr

