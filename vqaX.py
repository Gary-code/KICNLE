import os

os.environ['PYTHONPATH'] = '.'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from transformers import GPT2Tokenizer, AutoConfig, BertConfig, BertTokenizerFast
from transformers import AdamW, get_linear_schedule_with_warmup
import json
from cococaption.pycocotools.coco import COCO
from cococaption.pycocoevalcap.eval import COCOEvalCap
from PIL import Image
from accelerate import Accelerator
from models.gpt import GPT2LMHeadModel
from models.clip_vit import ImageEncoder
from models.oscar import BertImgModel
from utils.data_utils import *
from utils import data_utils
from utils.eval_utils import top_filtering
import h5py
from transformers import BertTokenizer, BertModel
from accelerate import InitProcessGroupKwargs

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

oscar_caption_model_path = './pre-trained/oscar/caption/pretrained_base/checkpoint-2000000'
oscar_tokenizer = BertTokenizerFast.from_pretrained(oscar_caption_model_path, do_lower_case=True)

with open('./objects_vocab.txt', 'r') as f:
    objects_vocab = f.read().splitlines()


def change_requires_grad(model, req_grad):
    for p in model.parameters():
        p.requires_grad = req_grad


def load_checkpoint(ckpt_path, epoch):
    model_name = 'nle_model_{}'.format(str(epoch))
    tokenizer_name = 'nle_gpt2_tokenizer_0'
    filename = 'ckpt_stats_' + str(epoch) + '.tar'

    tokenizer = GPT2Tokenizer.from_pretrained(ckpt_path + tokenizer_name)  # load tokenizer
    model = GPT2LMHeadModel.from_pretrained(ckpt_path + model_name).to(device)  # load model with config
    opt = torch.load(ckpt_path + filename)
    optimizer = get_optimizer(model, learning_rate)
    optimizer.load_state_dict(opt['optimizer_state_dict'])
    start_epoch = opt['epoch'] + 1
    scheduler_dic = opt['scheduler']
    del opt
    torch.cuda.empty_cache()

    return tokenizer, model, optimizer, scheduler_dic, start_epoch


def load_pretrained():
    model_path = 'pretrained_model/pretrain_model_11'
    tokenizer_path = 'pretrained_model/pretrain_tokenizer_0'
    # model_path = './Reasoning/KB-VCR/nlxgpt/pretrained_model/VQA-X/nle_model_11'
    # tokenizer_path = './Reasoning/KB-VCR/nlxgpt/pretrained_model/VQA-X/nle_gpt2_tokenizer_0'
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)  # load tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_path).to(device)  # load model with config
    return tokenizer, model


def save_checkpoint(epoch, unwrapped_model, optimizer, tokenizer, scheduler, ckpt_path, **kwargs):
    model_name = 'nle_model_{}'.format(str(epoch))
    tokenizer_name = 'nle_gpt2_tokenizer_{}'.format(str(epoch))
    filename = 'ckpt_stats_' + str(epoch) + '.tar'

    if epoch == 0:
        tokenizer.save_pretrained(ckpt_path + tokenizer_name)  # save tokenizer

    unwrapped_model.save_pretrained(ckpt_path + model_name, save_function=accelerator.save)

    opt = {'epoch': epoch,
           'optimizer_state_dict': optimizer.state_dict(),
           'scheduler': scheduler.state_dict(),
           **kwargs}

    accelerator.save(opt, ckpt_path + filename)


# def get_scores(annFile, resFile, save_scores_path):

#     coco = COCO(annFile)
#     cocoRes = coco.loadRes(resFile)
#     cocoEval = COCOEvalCap(coco, cocoRes)
#     cocoEval.evaluate()
#     with open(save_scores_path, 'w') as w:
#         json.dump(cocoEval.eval, w)


def filter_and_get_scores(resFileExp, save_scores_pathExp, full_predictions, exp_predictions, epoch):
    all_file = json.load(open(nle_data_test_path, 'r'))

    gt_answers = {}
    for key, value in all_file.items():
        gt_answers[int(key)] = data_utils.proc_ans(value['answers'])

    pred_answers = {}
    for item in full_predictions:
        pred_answers[item['image_id']] = item['caption'].split("because")[0].strip()

    correct_keys = []
    for key, value in pred_answers.items():
        gt_answer = gt_answers[key]
        # to measure accuracy for VQA, please change "==" to "in" (if value in gt_answer:)
        if value == gt_answer:
            correct_keys.append(key)

    print(f'acc: {len(correct_keys) / len(gt_answers)}')

    exp_preds = [item for item in exp_predictions if item['image_id'] in correct_keys]

    with open(resFileExp, 'w') as w:
        json.dump(exp_preds, w)

    coco = COCO(annFileExp)
    cocoRes = coco.loadRes(resFileExp)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()
    if epoch < 30:
        with open(save_scores_pathExp, 'w') as w:
            json.dump(cocoEval.eval, w)


class VQAXTrainDataset(Dataset):

    def __init__(self, path, transform, tokenizer, max_seq_len):
        self.kb = json.load(open(kb_train_pth, 'r'))
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_seq_len = max_seq_len  # question + <bos> The answer is <answer> becase <explanation> <eos>
        self.data = json.load(open(path, 'r'))
        self.ids_list = list(self.data.keys())
        self.qid_to_id = {str(qid): i for i, qid in enumerate(list(self.data.keys()))}
        for k, v in self.data.items():
            if len(v['explanation']) > 1:  # some questions have more than one explanation
                # duplicate them for loading. -1 because one explanation is already in ids_list
                self.ids_list += [str(k)] * (len(v['explanation']) - 1)

        self.index_tracker = {k: len(v['explanation']) - 1 for k, v in self.data.items()}

        # loading 36 objects
        self.image_features_path_36_obj_version = './pre-trained/faster-RCNN/genome-trainval.h5'
        self.coco_id_to_index = self._create_coco_id_to_index()
        self.coco_ids = [q['image_id'] for _, q in self.data.items()]

    def _create_coco_id_to_index(self):
        """ Create a mapping from a COCO image id into the corresponding index into the h5 file """
        with h5py.File(self.image_features_path_36_obj_version, 'r') as features_file:
            coco_ids = features_file['ids'][()]
        coco_id_to_index = {id: i for i, id in enumerate(coco_ids)}
        return coco_id_to_index

    def load_image_36_object_version(self, image_id):
        """ Load an image """
        if not hasattr(self, 'features_file'):
            # Loading the h5 file has to be done here and not in __init__ because when the DataLoader
            # forks for multiple works, every child would use the same file object and fail
            # Having multiple readers using different file objects is fine though, so we just init in here.
            self.features_file_36_obj_version = h5py.File(self.image_features_path_36_obj_version, 'r')
        index = self.coco_id_to_index[int(image_id)]
        img = self.features_file_36_obj_version['features'][index]
        bboxes = self.features_file_36_obj_version['boxes'][index]
        widths = self.features_file_36_obj_version['widths'][index]
        heights = self.features_file_36_obj_version['heights'][index]
        clses = self.features_file_36_obj_version['objects_id'][index]
        return img, bboxes, widths, heights, clses

    def __getitem__(self, i):
        quention_id = self.ids_list[i]
        sample = self.data[quention_id]
        idx = self.qid_to_id[quention_id]
        img_name = sample['image_name']
        img_id = sample['image_id']
        v, bboxes, _, _, clses = self.load_image_36_object_version(img_id)

        v = torch.from_numpy(v).float().T
        box = torch.from_numpy(bboxes).float().T
        box_w = box[:, 3] - box[:, 1]
        box_h = box[:, 2] - box[:, 0]
        v = torch.cat((v, box, box_w.unsqueeze(dim=1), box_h.unsqueeze(dim=1)), dim=-1)

        objects_repeat = [objects_vocab[cls] if ',' not in objects_vocab[cls] else objects_vocab[cls].split()[0] for cls
                          in
                          clses]

        # objects = []
        # for obj in objects_repeat:
        #     if obj in objects:
        #         continue
        #     objects.append(obj)

        # text_o = " ".join(objects)
        # max_object_len = 13
        # bert_objects_ids = bert_tokenizer(text_o, return_tensors="pt")['input_ids'][0, :]
        # if len(bert_objects_ids) > max_object_len:
        #     bert_objects_ids = bert_objects_ids[:max_object_len]
        # bert_objects_ids = torch.cat((bert_objects_ids, torch.tensor([0] * (max_object_len - len(bert_objects_ids))))).long()

        bert_objects_ids = clses
        text_a = data_utils.proc_ques(sample['question'])  # question <str>
        text_answer = data_utils.proc_ans(sample['answers'])

        exp_idx = self.index_tracker[
            quention_id]  # the index of the explanation for questions with multiple explanations
        if exp_idx > 0:
            self.index_tracker[quention_id] -= 1  # decrease usage

        text_b = sample['explanation'][exp_idx]  # explanation

        # tokenization process
        o_segment_id, k_segment_id, q_segment_id, a_segment_id, e_segment_id = self.tokenizer.convert_tokens_to_ids(
            ['<object>',
             '<knowledge>',
             '<question>',
             '<answer>',
             '<explanation>'])

        tokens = self.tokenizer.tokenize(text_a)
        labels = [-100] * len(tokens)  # we dont want to predict the question, set to pad to ignore in XE
        segment_ids = [q_segment_id] * len(tokens)

        if 'best_knowledge' in sample.keys():
            tokens_know = self.tokenizer.tokenize(" the knowledge is " + sample['best_knowledge'])
            tokens += tokens_know
            labels += [-100] * len(tokens_know)
            segment_ids += [k_segment_id] * len(tokens_know)

        # bert tokenize best knowledge
        if 'best_knowledge' in sample.keys():
            bert_knowledge_ids = bert_tokenizer(sample['best_knowledge'], return_tensors="pt")['input_ids'][0, :]
            bert_knowledge_mask = [1 for _ in bert_knowledge_ids]
            if len(bert_knowledge_ids) > knowledge_max_len:
                bert_knowledge_ids = bert_knowledge_ids[:knowledge_max_len]
                bert_knowledge_mask = bert_knowledge_mask[:knowledge_max_len]
            bert_knowledge_mask += [0] * (knowledge_max_len - len(bert_knowledge_ids))
            bert_knowledge_ids = torch.cat(
                (bert_knowledge_ids, torch.tensor([0] * (knowledge_max_len - len(bert_knowledge_ids))))).long()

        else:
            bert_knowledge_ids = \
            bert_tokenizer(bert_tokenizer.pad_token * (knowledge_max_len - 2), return_tensors="pt")['input_ids'][0, :]
            bert_knowledge_mask = [0] * (knowledge_max_len)
        bert_knowledge_mask = torch.tensor(bert_knowledge_mask, dtype=torch.long)

        # Q, K -> A, answer_init
        answer_init = [self.tokenizer.bos_token] + self.tokenizer.tokenize(" the answer is " + text_answer) + [
            self.tokenizer.eos_token]
        answer_init_len = len(answer_init)
        tokens_init_ans = tokens + answer_init
        segment_init_ans_ids = segment_ids + [a_segment_id] * answer_init_len
        labels_init_ans = labels + [-100] + answer_init[1:]

        if len(tokens_init_ans) > question_max_len + knowledge_max_len + answer_max_len:
            tokens_init_ans = tokens_init_ans[:question_max_len + knowledge_max_len + answer_max_len]
            labels_init_ans = labels_init_ans[:question_max_len + knowledge_max_len + answer_max_len]
            segment_init_ans_ids = segment_init_ans_ids[:question_max_len + knowledge_max_len + answer_max_len]

        assert len(tokens_init_ans) == len(segment_init_ans_ids)
        assert len(tokens_init_ans) == len(labels_init_ans)

        seq_len = len(tokens_init_ans)
        padding_len = question_max_len + knowledge_max_len + answer_max_len - seq_len
        tokens_init_ans = tokens_init_ans + ([self.tokenizer.pad_token] * padding_len)
        labels_init_ans = labels_init_ans + ([-100] * padding_len)

        segment_init_ans_ids += ([a_segment_id] * padding_len)
        input_ids_init_ans = self.tokenizer.convert_tokens_to_ids(tokens_init_ans)
        input_ids_init_ans = torch.tensor(input_ids_init_ans, dtype=torch.long)

        labels_init_ans = [self.tokenizer.convert_tokens_to_ids(t) if t != -100 else t for t in labels_init_ans]
        labels_init_ans = torch.tensor(labels_init_ans, dtype=torch.long)

        segment_init_ans_ids = torch.tensor(segment_init_ans_ids, dtype=torch.long)

        # Q, K, A -> E
        answer_to_ex = self.tokenizer.tokenize(" the answer is " + text_answer)
        answer_to_ex_len = len(answer_to_ex)
        explanation_to_ex = [self.tokenizer.bos_token] + self.tokenizer.tokenize(" because " + text_b) + [
            self.tokenizer.eos_token]
        explanation_to_ex_len = len(explanation_to_ex)

        segment_to_ex_ids = segment_ids + [a_segment_id] * answer_to_ex_len + [e_segment_id] * explanation_to_ex_len
        tokens_to_ex = tokens + answer_to_ex + explanation_to_ex

        labels_to_ex = labels + [-100] * answer_to_ex_len + [-100] + explanation_to_ex[1:]

        if len(tokens_to_ex) > self.max_seq_len:
            tokens_to_ex = tokens_to_ex[:self.max_seq_len]
            labels_to_ex = labels_to_ex[:self.max_seq_len]
            segment_to_ex_ids = segment_to_ex_ids[:self.max_seq_len]

        assert len(tokens_to_ex) == len(labels_to_ex)
        assert len(tokens_to_ex) == len(segment_to_ex_ids)

        seq_len = len(tokens_to_ex)
        padding_len = self.max_seq_len - seq_len
        tokens_to_ex = tokens_to_ex + ([self.tokenizer.pad_token] * padding_len)
        labels_to_ex = labels_to_ex + ([-100] * padding_len)

        segment_to_ex_ids += ([e_segment_id] * padding_len)
        input_ids_to_ex = self.tokenizer.convert_tokens_to_ids(tokens_to_ex)
        input_ids_to_ex = torch.tensor(input_ids_to_ex, dtype=torch.long)

        labels_to_ex = [self.tokenizer.convert_tokens_to_ids(t) if t != -100 else t for t in labels_to_ex]
        labels_to_ex = torch.tensor(labels_to_ex, dtype=torch.long)

        segment_to_ex_ids = torch.tensor(segment_to_ex_ids, dtype=torch.long)

        # Q, K, E -> A
        explanation_to_ans = self.tokenizer.tokenize(" because " + text_b)
        explanation_to_ans_len = len(explanation_to_ans)
        answer_to_ans = [self.tokenizer.bos_token] + self.tokenizer.tokenize(" so the answer is " + text_answer) + [
            self.tokenizer.eos_token]
        answer_to_ans_len = len(answer_to_ans)

        segment_to_ans_ids = segment_ids + [e_segment_id] * explanation_to_ans_len + [a_segment_id] * answer_to_ans_len
        tokens_to_ans = tokens + explanation_to_ans + answer_to_ans

        labels_to_ans = labels + [-100] * explanation_to_ans_len + [-100] + answer_to_ans[1:]

        if len(tokens_to_ans) > self.max_seq_len:
            tokens_to_ans = tokens_to_ans[:self.max_seq_len]
            labels_to_ans = labels_to_ans[:self.max_seq_len]
            segment_to_ans_ids = segment_to_ans_ids[:self.max_seq_len]

        assert len(tokens_to_ans) == len(labels_to_ans)
        assert len(tokens_to_ans) == len(segment_to_ans_ids)

        seq_len = len(tokens_to_ans)
        padding_len = self.max_seq_len - seq_len
        tokens_to_ans = tokens_to_ans + ([self.tokenizer.pad_token] * padding_len)
        labels_to_ans = labels_to_ans + ([-100] * padding_len)

        segment_to_ans_ids += ([a_segment_id] * padding_len)
        input_ids_to_ans = self.tokenizer.convert_tokens_to_ids(tokens_to_ans)
        input_ids_to_ans = torch.tensor(input_ids_to_ans, dtype=torch.long)

        labels_to_ans = [self.tokenizer.convert_tokens_to_ids(t) if t != -100 else t for t in labels_to_ans]
        labels_to_ans = torch.tensor(labels_to_ans, dtype=torch.long)

        segment_to_ans_ids = torch.tensor(segment_to_ans_ids, dtype=torch.long)

        # oscar part: Q and obj
        oscar_input_ids = oscar_tokenizer.encode(text_a)
        oscar_segment_ids = [0 for _ in oscar_input_ids]
        oscar_mask = [1 for _ in oscar_input_ids]
        q_len = len(oscar_input_ids)
        prefix_max_len = question_max_len + knowledge_max_len
        if q_len > prefix_max_len:
            oscar_input_ids = oscar_input_ids[:prefix_max_len]
            oscar_segment_ids = oscar_segment_ids[:prefix_max_len]
            oscar_mask = oscar_mask[:prefix_max_len]
        oscar_input_ids = oscar_input_ids + oscar_tokenizer.encode(
            ' '.join([oscar_tokenizer.pad_token] * (prefix_max_len - q_len)))[1:-1]
        oscar_segment_ids = oscar_segment_ids + ([0] * (prefix_max_len - q_len))
        oscar_mask = oscar_mask + ([0] * (prefix_max_len - q_len))

        oscar_input_ids = torch.tensor(oscar_input_ids, dtype=torch.long)
        oscar_segment_ids = torch.tensor(oscar_segment_ids, dtype=torch.long)
        oscar_mask = torch.tensor(oscar_mask, dtype=torch.long)

        oscar_mask = torch.cat((oscar_mask, torch.ones(v.shape[0], dtype=torch.long)), dim=0)

        # load image features
        coco_image_path = './VQA-Dataset'
        folder = coco_image_path + '/train2014/' if 'train' in img_name else coco_image_path + '/val2014/'
        img_path = folder + img_name
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        qid = torch.LongTensor([int(quention_id)])

        # return (v, img, qid, input_ids, labels, segment_ids, bert_objects_ids)
        # print([t.shape for t in  (v, img, qid, input_ids, labels, segment_ids, answer_ids, explanation_ids, bert_objects_ids)])
        # if bert_knowledge_mask.shape[0] != knowledge_max_len or bert_knowledge_ids.shape[0] != knowledge_max_len:
        #     print(bert_knowledge_mask.shape, bert_knowledge_ids.shape)
        # tmp = (v, oscar_input_ids, oscar_segment_ids, oscar_mask, img, qid, bert_objects_ids, bert_knowledge_ids, bert_knowledge_mask, input_ids_init_ans, labels_init_ans, segment_init_ans_ids, input_ids_to_ex, labels_to_ex, segment_to_ex_ids, input_ids_to_ans, labels_to_ans, segment_to_ans_ids)

        return (v, oscar_input_ids, oscar_segment_ids, oscar_mask, img, qid, bert_objects_ids, bert_knowledge_ids,
                bert_knowledge_mask, input_ids_init_ans, labels_init_ans, segment_init_ans_ids, input_ids_to_ex,
                labels_to_ex, segment_to_ex_ids, input_ids_to_ans, labels_to_ans, segment_to_ans_ids)

    def __len__(self):
        return len(self.ids_list)


class VQAXEvalDataset(Dataset):

    def __init__(self, path, transform, tokenizer, max_seq_len):
        self.kb = json.load(open(kb_test_pth, 'r'))
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_seq_len = max_seq_len  # question + <bos> The answer is <answer> becase <explanation> <eos>
        self.data = json.load(open(path, 'r'))
        self.ids_list = list(self.data.keys())
        # loading 36 objects
        self.image_features_path_36_obj_version = './pre-trained/faster-RCNN/genome-trainval.h5'
        self.coco_id_to_index = self._create_coco_id_to_index()
        self.coco_ids = [q['image_id'] for _, q in self.data.items()]

    def _create_coco_id_to_index(self):
        """ Create a mapping from a COCO image id into the corresponding index into the h5 file """
        with h5py.File(self.image_features_path_36_obj_version, 'r') as features_file:
            coco_ids = features_file['ids'][()]
        coco_id_to_index = {id: i for i, id in enumerate(coco_ids)}
        return coco_id_to_index

    def load_image_36_object_version(self, image_id):
        """ Load an image """
        if not hasattr(self, 'features_file'):
            # Loading the h5 file has to be done here and not in __init__ because when the DataLoader
            # forks for multiple works, every child would use the same file object and fail
            # Having multiple readers using different file objects is fine though, so we just init in here.
            self.features_file_36_obj_version = h5py.File(self.image_features_path_36_obj_version, 'r')
        index = self.coco_id_to_index[int(image_id)]
        img = self.features_file_36_obj_version['features'][index]
        bboxes = self.features_file_36_obj_version['boxes'][index]
        widths = self.features_file_36_obj_version['widths'][index]
        heights = self.features_file_36_obj_version['heights'][index]
        clses = self.features_file_36_obj_version['objects_id'][index]
        # clses = clses.split(';')
        return img, bboxes, widths, heights, clses

    def __getitem__(self, i):
        quention_id = self.ids_list[i]
        sample = self.data[quention_id]
        img_name = sample['image_name']
        img_id = sample['image_id']
        v, bboxes, _, _, clses = self.load_image_36_object_version(img_id)
        v = torch.from_numpy(v).float().T
        box = torch.from_numpy(bboxes).float().T
        box_w = box[:, 3] - box[:, 1]
        box_h = box[:, 2] - box[:, 0]
        v = torch.cat((v, box, box_w.unsqueeze(dim=1), box_h.unsqueeze(dim=1)), dim=-1)

        # objects_repeat = [objects_vocab[cls] if ',' not in objects_vocab[cls] else objects_vocab[cls].split()[0] for cls in
        #            clses]
        # objects = []
        # for obj in objects_repeat:
        #     if obj in objects:
        #         continue
        #     objects.append(obj)

        # text_o = " ".join(objects)
        # max_object_len = 13
        # bert_objects_ids = bert_tokenizer(text_o, return_tensors="pt")['input_ids'][0, :]
        # if len(bert_objects_ids) > max_object_len:
        #     bert_objects_ids = bert_objects_ids[:max_object_len]
        # bert_objects_ids = torch.cat((bert_objects_ids, torch.tensor([0] * (max_object_len - len(bert_objects_ids))))).long()

        bert_objects_ids = clses

        text_a = data_utils.proc_ques(sample['question'])  # question
        # tokenization process
        o_segment_id, k_segment_id, q_segment_id, a_segment_id, e_segment_id = self.tokenizer.convert_tokens_to_ids(
            ['<object>', '<knowledge>', '<question>', '<answer>', '<explanation>'])

        tokens = self.tokenizer.tokenize(text_a)
        segment_ids = [q_segment_id] * len(tokens)

        if 'best_knowledge' in sample.keys():
            tokens_know = self.tokenizer.tokenize(" the knowledge is " + sample['best_knowledge'])
            tokens += tokens_know
            segment_ids += [k_segment_id] * len(tokens_know)

        if 'best_knowledge' in sample.keys():
            bert_knowledge_ids = bert_tokenizer(sample['best_knowledge'], return_tensors="pt")['input_ids'][0, :]
            bert_knowledge_mask = [1 for _ in bert_knowledge_ids]
            if len(bert_knowledge_ids) > knowledge_max_len:
                bert_knowledge_ids = bert_knowledge_ids[:knowledge_max_len]
                bert_knowledge_mask = bert_knowledge_mask[:knowledge_max_len]
            bert_knowledge_mask += [0] * (knowledge_max_len - len(bert_knowledge_ids))
            bert_knowledge_ids = torch.cat(
                (bert_knowledge_ids, torch.tensor([0] * (knowledge_max_len - len(bert_knowledge_ids))))).long()


        else:
            bert_knowledge_ids = \
            bert_tokenizer(bert_tokenizer.pad_token * (knowledge_max_len - 2), return_tensors="pt")['input_ids'][0, :]
            bert_knowledge_mask = [0] * (knowledge_max_len)
        bert_knowledge_mask = torch.tensor(bert_knowledge_mask, dtype=torch.long)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)

        coco_image_path = './VQA-Dataset'
        folder = coco_image_path + '/train2014/' if 'train' in img_name else coco_image_path + '/val2014/'

        # oscar part: Q and obj
        oscar_input_ids = oscar_tokenizer.encode(text_a)
        oscar_segment_ids = [0 for _ in oscar_input_ids]
        oscar_mask = [1 for _ in oscar_input_ids]
        q_len = len(oscar_input_ids)
        prefix_max_len = question_max_len + knowledge_max_len
        if q_len > prefix_max_len:
            oscar_input_ids = oscar_input_ids[:prefix_max_len]
            oscar_segment_ids = oscar_segment_ids[:prefix_max_len]
            oscar_mask = oscar_mask[:prefix_max_len]
        oscar_input_ids = oscar_input_ids + oscar_tokenizer.encode(
            ' '.join([oscar_tokenizer.pad_token] * (prefix_max_len - q_len)))[1:-1]
        oscar_segment_ids = oscar_segment_ids + ([0] * (prefix_max_len - q_len))
        oscar_mask = oscar_mask + ([0] * (prefix_max_len - q_len))

        oscar_input_ids = torch.tensor(oscar_input_ids, dtype=torch.long)
        oscar_segment_ids = torch.tensor(oscar_segment_ids, dtype=torch.long)
        oscar_mask = torch.tensor(oscar_mask, dtype=torch.long)

        oscar_mask = torch.cat((oscar_mask, torch.ones(v.shape[0], dtype=torch.long)), dim=0)

        # folder = 'images/train2014/' if 'train' in img_name else 'images/val2014/'   # test and val are both in val2014
        img_path = folder + img_name
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        qid = torch.LongTensor([int(quention_id)])

        return (v, oscar_input_ids, oscar_segment_ids, oscar_mask, img, qid, input_ids, segment_ids, bert_objects_ids,
                bert_knowledge_ids, bert_knowledge_mask)

    def __len__(self):
        return len(self.ids_list)


def sample_sequences(model, tokenizer, loader):
    model.eval()
    results_exp = []
    results_full = []
    SPECIAL_TOKENS = ['<|endoftext|>', '<pad>', '<object>', '<knowledge>', '<question>', '<answer>', '<explanation>']
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    because_token = tokenizer.convert_tokens_to_ids('Ġbecause')
    max_len = 20

    for i, batch in enumerate(loader):

        current_output = []
        batch = tuple(input_tensor.to(device) for input_tensor in batch)
        obj, oscar_input_ids, oscar_segment_ids, oscar_mask, img, img_id, input_ids, segment_ids, objects_id, bert_knowledge_ids, bert_knowledge_mask = batch
        # oscar_ouput = oscar_model(oscar_input_ids, img_feats=obj, attention_mask=oscar_mask, position_ids=None, token_type_ids=oscar_segment_ids, head_mask=None)

        img_embeddings, emb_mask = image_encoder(img, oscar_input_ids, bert_knowledge_mask, objects_id)
        #
        # hidden_state = torch.cat((img_embeddings, oscar_ouput[0]), dim=1)
        hidden_state = img_embeddings

        always_exp = False

        with torch.no_grad():
            outputs = model._sample(obj,
                                    oscar_input_ids,
                                    oscar_segment_ids,
                                    oscar_mask,
                                    input_ids=input_ids,
                                    past_key_values=None,
                                    attention_mask=None,
                                    token_type_ids=segment_ids,
                                    position_ids=None,
                                    encoder_hidden_states=hidden_state,
                                    encoder_attention_mask=emb_mask,
                                    labels=None,
                                    use_cache=False,
                                    return_dict=True)

            current_output = outputs

            # for step in range(max_len + 1):

            #     if step == max_len:
            #         break

            #     outputs = model(input_ids=input_ids,
            #                     past_key_values=None,
            #                     attention_mask=None,
            #                     token_type_ids=segment_ids,
            #                     position_ids=None,
            #                     encoder_hidden_states=img_embeddings,
            #                     encoder_attention_mask=None,
            #                     labels=None,
            #                     use_cache=False,
            #                     return_dict=True)

            #     lm_logits = outputs.logits
            #     logits = lm_logits[0, -1, :] / temperature
            #     logits = top_filtering(logits, top_k=top_k, top_p=top_p)
            #     probs = F.softmax(logits, dim=-1)
            #     prev = torch.topk(probs, 1)[1] if no_sample else torch.multinomial(probs, 1)

            #     if prev.item() in special_tokens_ids:
            #         break

            #     # take care of when to start the <explanation> token
            #     if not always_exp:

            #         if prev.item() != because_token:
            #             new_segment = special_tokens_ids[-2]  # answer segment
            #         else:
            #             new_segment = special_tokens_ids[-1]  # explanation segment
            #             always_exp = True
            #     else:
            #         new_segment = special_tokens_ids[-1]  # explanation segment

            #     new_segment = torch.LongTensor([new_segment]).to(device)
            #     current_output.append(prev.item())
            #     input_ids = torch.cat((input_ids, prev.unsqueeze(0)), dim=1)
            #     segment_ids = torch.cat((segment_ids, new_segment.unsqueeze(0)), dim=1)

        decoded_sequences = tokenizer.decode(current_output, skip_special_tokens=True).lstrip()
        # print(decoded_sequences)
        results_full.append({"image_id": img_id.item(), "caption": decoded_sequences})

        if 'because' in decoded_sequences:
            cut_decoded_sequences = decoded_sequences.split('because')[-1].strip()
        else:
            cut_decoded_sequences = " ".join(decoded_sequences.split()[2:])

        results_exp.append({"image_id": img_id.item(), "caption": cut_decoded_sequences})
        print("\rEvaluation: Finished {}/{}".format(i, len(loader)), end='          ')

    return results_full, results_exp


def get_optimizer(model, learning_rate):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    return optimizer


accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs()])
device = accelerator.device

finetune_pretrained = True  # if True, finetunes from the image captioning model
eval_batch_size = 1
img_size = 224
base_dataset_path = './Reasoning/Dataset/'
ckpt_path = 'ckpts/VQA-X/'
# caption_save_path = 'cococaption/results/VQA-X/'
annFileExp = 'cococaption/annotations/vqaX_test_annot_exp.json'
# annFileFull = 'cococaption/annotations/vqaX_test_annot_full.json'
# nle_data_train_path = base_dataset_path + 'VQA-X/vqaX_train.json'
# nle_data_test_path = base_dataset_path + 'VQA-X/vqaX_test.json'
# nle_data_val_path = base_dataset_path + 'VQA-X/vqaX_val.json'

# knowledge version
caption_save_path = 'cococaption/results/VQA-X-KB/'
annFileExp = 'cococaption/annotations/vqaX_test_annot_exp.json'
nle_data_train_path = base_dataset_path + 'VQA-X/vqaX_train_knowledge_v3.json'
nle_data_test_path = base_dataset_path + 'VQA-X/vqaX_test_knowledge_v3.json'

kb_train_pth = './kb_data/VQA-X/vqaX_train_kb.json'
kb_test_pth = './kb_data/VQA-X/vqaX_test_kb.json'
kb_val_pth = './kb_data/VQA-X/vqaX_train_kb.json'
max_seq_len = 40
load_from_epoch = None
no_sample = True
top_k = 0
top_p = 0.9
batch_size = 62  # per GPU
num_train_epochs = 50
weight_decay = 0
learning_rate = 2e-5 if not finetune_pretrained else 1e-5
# learning_rate = 2e-5
gradient_accumulation_steps = 1
start_epoch = 0
temperature = 1
explanation_max_len = 20
answer_max_len = 3
question_max_len = 10
knowledge_max_len = 8

# image_encoder = ImageEncoder(device).to(device)
# change_requires_grad(image_encoder, False)

if load_from_epoch is not None:
    tokenizer, model, optimizer, scheduler_dic, start_epoch = load_checkpoint(ckpt_path, load_from_epoch)

else:

    if finetune_pretrained:
        tokenizer, model = load_pretrained()
        optimizer = get_optimizer(model, learning_rate)
        tokenizer.add_special_tokens({'additional_special_tokens': ['<knowledge>']})
        model.resize_token_embeddings(len(tokenizer))

    else:
        tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
        orig_num_tokens = len(tokenizer.encoder)

        num_new_tokens = tokenizer.add_special_tokens({'pad_token': '<pad>',
                                                       'additional_special_tokens': ['<knowledge>', '<question>',
                                                                                     '<answer>', '<explanation>']})

        assert len(tokenizer) == orig_num_tokens + num_new_tokens
        config = AutoConfig.from_pretrained('distilgpt2')

        # Add configs
        setattr(config, 'img_size', None)
        setattr(config, 'max_seq_len', None)
        config.img_size = img_size
        config.max_seq_len = max_seq_len
        config.add_cross_attention = True

        model = GPT2LMHeadModel.from_pretrained('distilgpt2', config=config)
        model.resize_token_embeddings(len(tokenizer))
        model = model.to(device)
        optimizer = get_optimizer(model, learning_rate)

print("Model Setup Ready...")

img_transform = transforms.Compose([transforms.Resize((img_size, img_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

train_dataset = VQAXTrainDataset(path=nle_data_train_path,
                                 transform=img_transform,
                                 tokenizer=tokenizer,
                                 max_seq_len=max_seq_len)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           pin_memory=True,
                                           num_workers=20)

# val_dataset = VQAXEvalDataset(path = nle_data_val_path,
#                               transform = img_transform,
#                               tokenizer = tokenizer,
#                               max_seq_len = max_seq_len)


# val_loader = torch.utils.data.DataLoader(val_dataset,
#                                          batch_size = 1,
#                                          shuffle=False,
#                                          pin_memory=True)

test_dataset = VQAXEvalDataset(path=nle_data_test_path,
                               transform=img_transform,
                               tokenizer=tokenizer,
                               max_seq_len=max_seq_len)

test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=1,
                                          shuffle=False,
                                          pin_memory=True,
                                          num_workers=20)

model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

t_total = (len(train_loader) // gradient_accumulation_steps) * num_train_epochs
warmup_steps = 0  # 0.10 * t_total
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

if load_from_epoch is not None:
    scheduler.load_state_dict(scheduler_dic)

for epoch in range(start_epoch, num_train_epochs):

    model.train()
    accum_loss = 0

    for step, batch in enumerate(train_loader):

        batch = tuple(input_tensor.to(device) for input_tensor in batch)
        # return (v, img, qid, input_ids, labels, segment_ids, answer_ids, explanation_ids, bert_objects_ids)

        obj, oscar_input_ids, oscar_segment_ids, oscar_mask, img, _, bert_obj_token, bert_knowledge_ids, bert_knowledge_mask, input_ids_init_ans, labels_init_ans, segment_init_ans_ids, input_ids_to_ex, labels_to_ex, segment_to_ex_ids, input_ids_to_ans, labels_to_ans, segment_to_ans_ids = batch

        img_embeddings, emb_mask = image_encoder(img, oscar_input_ids, bert_knowledge_mask, bert_obj_token)

        # hidden_state = torch.cat((img_embeddings, oscar_ouput[0]), dim=1)
        # hidden_state = img_embeddings
        outputs = model(obj,
                        oscar_input_ids,
                        oscar_segment_ids,
                        oscar_mask,
                        input_ids_init_ans,
                        labels_init_ans,
                        segment_init_ans_ids,
                        input_ids_to_ex,
                        labels_to_ex,
                        segment_to_ex_ids,
                        input_ids_to_ans,
                        labels_to_ans,
                        segment_to_ans_ids,
                        past_key_values=None,
                        attention_mask=None,
                        position_ids=None,
                        encoder_hidden_states=img_embeddings,
                        encoder_attention_mask=emb_mask,
                        use_cache=False,
                        return_dict=True)

        # V, Q, K
        loss = outputs.loss
        loss = loss / gradient_accumulation_steps
        accelerator.backward(loss)
        accum_loss += loss.item()

        # torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 0.5)

        if step % gradient_accumulation_steps == 0 or step == len(train_loader) - 1:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            accelerator.print("\rEpoch {} / {}, Iter {} / {}, Loss: {:.3f}".format(epoch,
                                                                                   num_train_epochs,
                                                                                   step, len(train_loader),
                                                                                   accum_loss),
                              end='          ')
            accum_loss = 0

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    # save_checkpoint(epoch, unwrapped_model, optimizer, tokenizer, scheduler, ckpt_path)

    # if accelerator.is_main_process:
    if True:
        results_full, results_exp = sample_sequences(unwrapped_model, tokenizer, test_loader)

        resFileExp = caption_save_path + 'captions_exp_' + str(epoch) + '.json'
        unf_resFileExp = caption_save_path + 'unf_captions_exp_' + str(epoch) + '.json'
        unf_resFileFull = caption_save_path + 'unf_captions_full_' + str(epoch) + '.json'
        save_scores_pathExp = caption_save_path + 'scores_exp_' + str(epoch) + '.json'

        if epoch < 30:
            with open(unf_resFileExp, 'w') as w:
                json.dump(results_exp, w)

            with open(unf_resFileFull, 'w') as w:
                json.dump(results_full, w)

        # unfiltered results
        # get_scores(annFileExp, unf_resFileExp, save_scores_pathExp)

        # filtered results
        filter_and_get_scores(resFileExp, save_scores_pathExp, results_full, results_exp, epoch)
