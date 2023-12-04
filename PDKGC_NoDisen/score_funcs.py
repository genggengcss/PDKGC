import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvE(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs

        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(self.configs.num_filt)
        self.bn2 = torch.nn.BatchNorm1d(self.configs.embed_dim)

        self.hidden_drop = torch.nn.Dropout(self.configs.hid_drop2)
        self.feature_drop = torch.nn.Dropout(self.configs.feat_drop)
        self.m_conv1 = torch.nn.Conv2d(1, out_channels=self.configs.num_filt, kernel_size=(self.configs.ker_sz, self.configs.ker_sz),
                                       stride=1, padding=0, bias=self.configs.bias)

        flat_sz_h = int(2 * self.configs.k_w) - self.configs.ker_sz + 1
        flat_sz_w = self.configs.k_h - self.configs.ker_sz + 1
        self.flat_sz = flat_sz_h * flat_sz_w * self.configs.num_filt
        self.fc = torch.nn.Linear(self.flat_sz, self.configs.embed_dim)

    def concat(self, e1_embed, rel_embed):
        e1_embed = e1_embed.view(-1, 1, e1_embed.size(-1))
        rel_embed = rel_embed.view(-1, 1, rel_embed.size(-1))
        stack_inp = torch.cat([e1_embed, rel_embed], 1)
        stack_inp = torch.transpose(stack_inp, 2, 1).reshape((-1, 1, 2 * self.configs.k_w, self.configs.k_h))
        return stack_inp

    def forward(self, head_emb, rel_emb):
        head_emb = head_emb.view(-1, self.configs.embed_dim)
        rel_emb = rel_emb.view(-1, self.configs.embed_dim)
        stk_inp = self.concat(head_emb, rel_emb)
        x = self.bn0(stk_inp)
        x = self.m_conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_drop(x)
        x = x.view(-1, self.flat_sz)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        # score = torch.sigmoid(x)
        x = x.view(-1, self.configs.num_factors, self.configs.embed_dim)
        return x

    def get_logits(self, pred, ent_embed, bias):
        logits = torch.einsum('bkf,nkf->bkn', [pred, ent_embed])
        logits += bias.expand_as(logits)
        return logits

class TransE(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs

        gamma_init = torch.FloatTensor([self.configs.init_gamma])
        if not self.configs.fix_gamma:
            self.register_parameter('gamma', nn.Parameter(gamma_init))

    def forward(self, head_emb, rel_emb):
        return head_emb + rel_emb

    def get_logits(self, pred, ent_embed, bias):
        if self.configs.gamma_method == 'ada':
            logits= self.gamma - torch.norm(pred.unsqueeze(1) - ent_embed, p=1, dim=3).transpose(1, 2)
        elif self.configs.gamma_method == 'norm':
            x = torch.sum(pred * pred, dim=-1)
            y = torch.sum(ent_embed * ent_embed, dim=-1)
            xy = torch.einsum('bkf,nkf->bkn', [pred, ent_embed])
            logits= self.gamma - (x.unsqueeze(2) + y.t() -  2 * xy)

        elif self.configs.gamma_method == 'fix':
            logits= self.configs.gamma - torch.norm(pred.unsqueeze(1) - ent_embed, p=1, dim=3).transpose(1, 2)
        return logits
    
class DistMult(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs

    def forward(self, head_emb, rel_emb):
        return head_emb * rel_emb

    def get_logits(self, pred, ent_embed, bias):
        logits = torch.einsum('bkf,nkf->bkn', [pred, ent_embed])
        logits += bias.expand_as(logits)
        return logits
