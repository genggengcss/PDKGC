from helper import *
import torch.nn as nn
from transformers import AutoConfig, AutoModel




class MEM_Model(nn.Module):
    def __init__(self, params, device):
        super(MEM_Model, self).__init__()
        self.p = params
        # self.embed_dim = self.p.embed_dim
        self.device = device
        self.plm_configs = AutoConfig.from_pretrained(self.p.pretrained_model)
        self.plm = AutoModel.from_pretrained(self.p.pretrained_model, config=self.plm_configs)


        # self.ent_classifier = torch.nn.Linear(self.plm_configs.hidden_size, self.p.num_ent)
        for p in self.plm.parameters():
            p.requires_grad = False

        self.loss_fn = get_loss_fn(params)

        ent_text_embeds_file = '../data/{}/{}.pt'.format(self.p.dataset, 'entity_bert_embeds')
        self.ent_text_embeds = torch.load(ent_text_embeds_file).to(self.device)
        self.ent_transform = torch.nn.Linear(self.plm_configs.hidden_size, self.plm_configs.hidden_size)



    def forward(self, sub, rel, text_ids, text_mask):


        output = self.plm(input_ids=text_ids, attention_mask=text_mask)
        last_hidden_state = output.last_hidden_state


        # mask_token_state = last_hidden_state[:, -2]
        # output = self.ent_classifier(mask_token_state)

        mask_token_state = last_hidden_state[:, -2]
        # output = self.ent_classifier(mask_token_state)
        output_tmp = self.ent_transform(mask_token_state)

        output = torch.einsum('bf,nf->bn', [output_tmp, self.ent_text_embeds])

        return output

