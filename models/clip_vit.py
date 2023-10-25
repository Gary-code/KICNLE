import clip
import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm

import pickle
import numpy as np

from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings, BertModel


with open('./objects_vocab.txt', 'r', encoding = 'utf-8') as f:
    entities = [entity.rstrip('\n') for entity in f.readlines()]
    
entities = [objects_vocab if ',' not in objects_vocab else objects_vocab.split()[0] for objects_vocab in entities]

class QuestionAttention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, question_dim, label_dim, dropout=0.5):
        """
        :param features_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(QuestionAttention, self).__init__()
        self.labels_att = weight_norm(nn.Linear(label_dim, question_dim))  # linear layer to transform decoder's output [b,topk,400]
        self.full_att = weight_norm(nn.Linear(question_dim, 1))  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, label_features, question_features):
        """
        Forward propagation.
        :param image_features: encoded images, a tensor of dimension (batch_size, 36, features_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.labels_att(label_features)  # (N, question_dim)
        att2 = question_features
        #print(att1.size())
        #print(att2.size())
        #print('--------------------------------------------------------')
        att = self.full_att(self.dropout(self.relu(att1 + att2.unsqueeze(0))))
        #print(att.size())
        alpha = self.softmax(att)  # (N, 1)
        #print(alpha.size())
        
        attention_weighted_encoding = (att1 * alpha).sum(dim=0)  # (question_dim)

        
        return attention_weighted_encoding


class ImageEncoder(nn.Module):

    def __init__(self, device):
        super(ImageEncoder, self).__init__()
        self.encoder, _ = clip.load("ViT-B/16", device=device)   # loads already in eval mode
        self.attention_dim = 768
        # self.encoder, _ = clip.load("ViT-L/14", device=device)   # loads already in eval mode
        # self.dropout = nn.Dropout(p=0.5)
        # self.linear = nn.Linear(2048, 768)
        self.bert_embedding = BertEmbeddings(BertConfig())
        oscar_caption_model_path = './pre-trained/oscar/caption/pretrained_base/checkpoint-2000000'
        self.q_k_attention = QuestionAttention(self.attention_dim, 400)
        self.bert = BertModel.from_pretrained(oscar_caption_model_path)

    def forward(self, x, bert_token=None, bert_mask=None, objects=None):
        """
        x: image
        Expects a tensor of size (batch_size, 3, 224, 224)
        """
        if bert_token != None:
            q_pooler = self.bert(bert_token)["pooler_output"]
            # KG2E
            kb_emb = []
            with open('/home/cike/Reasoning/KB-VCR/Trans-Implementation/sample/vqa_sample_embeding.pkl', 'rb') as pickle_file:
                kb_emb = pickle.load(pickle_file)
                
            # text_embedding = self.bert(bert_token)[0]
            mat_2 = []
            knowledge_mask = torch.ones_like(objects).long().to(objects.device)
            for i, sample_objs in enumerate(objects):
                mat_1 = []
                for j, obj in enumerate(sample_objs):
                    kb_emb_m = [[float(i) for i in j] for j in kb_emb[kb_emb[:, 0] == entities[int(obj)]][:, -1]]
                    if len(kb_emb_m) == 0:
                        knowledge_mask[i, j] = 0
                        mat_1.append(torch.zeros(self.attention_dim))
                        continue
                    kb_emb_m = np.array(kb_emb_m)
                    kb_mid = self.q_k_attention(torch.tensor(kb_emb_m).float().cuda(), q_pooler[i, j])
                    mat_1.append(kb_mid)
                
                val = torch.tensor([item.cpu().detach().numpy() for item in mat_1]).cuda() 
                mat_2.append(val)

            knowledge_emb = torch.tensor([item.cpu().detach().numpy() for item in mat_2]).cuda()


        # obj_emb = self.linear(obj)
        # if bert_token != None:
        #     text_embedding = self.bert_embedding(bert_token)
            # text_embedding = self.bert(bert_token)[0]
        x = x.type(self.encoder.visual.conv1.weight.dtype)
        x = self.encoder.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.encoder.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.encoder.visual.positional_embedding.to(x.dtype)
        x = self.encoder.visual.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.encoder.visual.transformer(x)
        grid_feats = x.permute(1, 0, 2)  # LND -> NLD    (N, 197, 768)
        grid_feats = self.encoder.visual.ln_post(grid_feats[:,1:])
        grid_mask = torch.ones_like(grid_feats[:,:, 0]).long().to(grid_feats.device)
        if bert_token != None:
            feat_mask = torch.cat((knowledge_mask, grid_mask), dim=1)
            return torch.cat((knowledge_emb, grid_feats.float()), dim=1), feat_mask
        else:
            return grid_feats.float(), grid_mask
        # return knowledge_emb, knowledge_mask
        # return obj_emb.float()
