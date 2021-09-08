# Author: StevenChaoo
# -*- coding:UTF-8 -*-


import sys
sys.path.extend(["../utils"])

import os
import json
import torch
import argparse
import logging

from tqdm import tqdm
from torch import nn
from torchcrf import CRF
from torch.optim import Adam
from transformers import BertForTokenClassification, BertConfig, BertModel
from transformers import AdamW
from transformers import BertTokenizer
from utils.dataset import NERDataset, NERDataLoader
from functools import reduce
from utils.functions import *
from utils.FGM import FGM


class BERTCRF(torch.nn.Module):
    def __init__(self, bert, d_model, num_tags, dropout=0.0, use_lstm=True):
        super().__init__()
        self.bert = bert
        self.use_lstm = use_lstm
        if self.use_lstm:
            self.lstm = nn.LSTM(input_size=d_model,
                                hidden_size=d_model,
                                batch_first=True,
                                bidirectional=True)
            self.classifier = torch.nn.Linear(d_model*2, num_tags)
        else:
            self.classifier = torch.nn.Linear(d_model, num_tags)
        self.crf = CRF(num_tags=num_tags,
                       batch_first=True)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, input_ids, input_tags, mask):
        x = self.bert(input_ids, mask)[0].float()
        x = self.dropout1(x)
        if self.use_lstm:
            x, _ = self.lstm(x)
        emissions = self.dropout2(self.classifier(x))
        loss = - self.crf(emissions, tags=input_tags, mask=mask,
                          reduction='token_mean')
        return loss

    def decode(self, input_ids, mask):
        x = self.bert(input_ids)[0].float()
        if self.use_lstm:
            x, _ = self.lstm(x)
        emissions = self.classifier(x)
        return self.crf.decode(emissions, mask)

    def trained_parameters(self):
        for name, param in self.named_parameters():
            if not name.startswith('bert'):
                yield param

    def save(self, tag2id, tokenizer, path):
        root = path
        if not os.path.exists(root):
            os.mkdir(root)
        torch.save(model, os.path.join(root, 'BERTCRF.torch'))
        with open(os.path.join(root, 'tag2id.json'), 'w') as f:
            json.dump(tag2id, f)
        tokenizer.save_pretrained(path)


def sent_truncate(batch, max_sent_len=512):
    sent_len = batch.input_ids.size(1)
    start, end = 0, 512
    while sent_len >= end:
        yield Batch(batch.input_ids[:, start: end], batch.input_tags[:, start: end])
        start += max_sent_len
        end += max_sent_len


def run_epoch(model, tokenizer, trainset_loader, device, batch_size, work_batch_size, lr=[5e-5, 5e-3]):
    opt = AdamW([{'params': model.bert.parameters(), 'lr': lr[0]}, {'params': model.trained_parameters(), 'lr': lr[1]}])
    if args.fgm:
        fgm = FGM(model)
    count = batch_size // work_batch_size
    for batch in trainset_loader.traversal(device):
        batch.to(device)
        input_ids = batch.input_ids
        input_tags = batch.input_tags
        attention_mask = batch.mask
        loss = model(input_ids=input_ids, input_tags=input_tags, mask=attention_mask)
        loss.backward()
        if args.fgm:
            fgm.attack()
            loss_adv = model(input_ids=input_ids, input_tags=input_tags, mask=attention_mask)
            loss_adv.backward()
            fgm.restore()
        count -= 1
        if count == 0:
            opt.step()
            opt.zero_grad()
            count = batch_size // work_batch_size


def test(model, dataset_loader, id2tag, device='cpu', epoch=0):
    total_target_ners = []
    total_predict_ners = []
    total_correct_ners = []
    for i, batch in enumerate(dataset_loader.traversal(device)):
        with torch.no_grad():
            predict_labels = model.decode(batch.input_ids, batch.mask)
            max_seq_len = max([len(seq) for seq in predict_labels])
            predict_labels = [seq + [0] * (max_seq_len - len(seq)) for seq in predict_labels]
            predict_labels = torch.tensor(predict_labels)
            target_ners, predict_ners, correct_ners = entity_match(epoch, batch.input_tags, predict_labels, batch.mask, id2tag)
            total_target_ners.extend(target_ners)
            total_predict_ners.extend(predict_ners)
            total_correct_ners.extend(correct_ners)
    precision = (len(total_correct_ners) + 1) / (len(total_predict_ners) + 1)
    recall = (len(total_correct_ners) + 1) / (len(total_target_ners) + 1)
    f_measure = 2 * precision * recall / (precision + recall)
    print('p={:.3%} r={:.3%} f={:.3%}'.format(precision, recall, f_measure))
    print("\n")
    label_evaluate(total_target_ners, total_predict_ners, total_correct_ners)
    return precision, recall, f_measure


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'finetune'])
    parser.add_argument('--pretrained_bert_path', default='data/pretrained_model/bert_wwm')
    parser.add_argument('--model_path', default='saved_model/ccks-bert')
    parser.add_argument('--saved_path', default='saved_model/ccks-bert')
    parser.add_argument('--trainset_path', default='["data/ccks-data-9-30/ccks_data0/2020train_up.txt"]')
    parser.add_argument('--testset_path', default='data/ccks-data-9-30/ccks_data0/2020test_up.txt')
    # parser.add_argument('--lr', default="[3e-5, 5e-4]")
    parser.add_argument('--lr', default="[1e-5, 1e-4]")
    parser.add_argument('--dropout', default=0.3, type=float)
    parser.add_argument('--cuda', default='1')
    parser.add_argument('--batch_size', default='8', type=int)
    parser.add_argument('--work_batch_size', default='-1', type=int)
    parser.add_argument('--use_lstm', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--visible_tags', default=None)
    parser.add_argument('--invisible_tags', default=None)
    parser.add_argument('--hidden_tags', default=None)
    parser.add_argument('--warmup')
    parser.add_argument('--epoches', default=50, type=int)
    parser.add_argument('--fgm', default=True)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    # Set cuda
    if args.cuda == 'cpu':
        cuda = args.cuda
    else:
        cuda = torch.device('cuda:' + args.cuda)
    
    # Some arguments not work in this work
    if args.work_batch_size == -1:
        args.work_batch_size = args.batch_size
    if args.visible_tags is not None:
        args.visible_tags = json.loads(args.visible_tags)
    if args.invisible_tags is not None:
        args.invisible_tags = json.loads(args.invisible_tags)
    if args.hidden_tags is not None:
        args.hidden_tags = json.loads(args.hidden_tags)

    # Load training set paths
    args.trainset_path = json.loads(args.trainset_path)

    # Load learning rate list
    args.lr = json.loads(args.lr)

    # Get tags
    tags = set()
    for dp in args.trainset_path:
        tags.update(extract_tags_from_dataset(dp))
    print('get tags from dataset.')

    # Add another tags
    if args.visible_tags is not None:
        for tag in tags.copy():
            if len(tag.split('-')) == 2 and tag.split('-')[1] not in args.visible_tags:
                tags.remove(tag)
    if args.invisible_tags is not None:
        for tag in tags.copy():
            if len(tag.split('-')) == 2 and tag.split('-')[1] in args.invisible_tags:
                tags.remove(tag)
    args.visible_tags = tags.copy()
    if args.hidden_tags is not None:
        for tag in tags.copy():
            if len(tag.split('-')) == 2 and tag.split('-')[1] in args.hidden_tags:
                tags.remove(tag)

    # Set tag2id and id2tag map
    tag2id = {tag: idx for idx, tag in enumerate(tags)}
    id2tag = {id_: tag for tag, id_ in tag2id.items()}

    # Train mode
    if args.mode == 'train':
        # Set basic configurations
        tokenizer = BertTokenizer.from_pretrained(args.pretrained_bert_path)
        config = BertConfig.from_pretrained(args.pretrained_bert_path)
        config.update({'num_labels': len(tag2id)})
        bert_model = BertModel.from_pretrained(args.pretrained_bert_path, config=config)

        # Initialize model
        model = BERTCRF(
                bert_model,
                d_model=config.hidden_size,
                num_tags=config.num_labels,
                dropout=args.dropout,
                use_lstm=args.use_lstm
                )

        # Start train
        model.train()
        model.to(cuda)

    # Finetune mode
    elif args.mode == 'finetune':
        tokenizer = BertTokenizer.from_pretrained(args.pretrained_bert_path)
        config = BertConfig.from_pretrained(args.pretrained_bert_path)
        model = torch.load(os.path.join(args.model_path, 'BERTCRF.torch'))
        model.to(cuda)
        with open(os.path.join(args.model_path, 'tag2id.json')) as f:
            tag2id_model = json.load(f)
        tags_model = set(tag2id_model)
        tags_dataset = set(tag2id)
        if len(tags_model - tags_dataset) > 0:
            print(f"some tags don't appears in dataset: {tags_model-tags_dataset}")
        if len(tags_dataset - tags_model) > 0:
            raise Exception(f"some new tags appears: {tags_dataset-tags_model}")
        tag2id = tag2id_model
        id2tag = {id_: tag for tag, id_ in tag2id.items()}


    # Load train dataset
    trainset = reduce(lambda x, y: x + y, (NERDataset(dp, visible_tags=args.visible_tags, hidden_tags=args.hidden_tags) for dp in args.trainset_path))
    trainset_loader = NERDataLoader(trainset, args.work_batch_size, tokenizer, tag2id)

    # Load test dataset
    testset = NERDataset(args.testset_path, visible_tags=args.visible_tags, hidden_tags=args.hidden_tags)
    testset_loader = NERDataLoader(testset, args.work_batch_size, tokenizer, tag2id)


    max_fmeasure = 0
    max_epoch = 0
    for i in range(args.epoches):
        print("========== Epoch {}/{} ==========".format(i, args.epoches))
        model.train()
        run_epoch(model, tokenizer, trainset_loader, cuda, args.batch_size, args.work_batch_size, lr=args.lr)
        model.eval()
        p, r, f = test(model, testset_loader, id2tag, cuda, i)
        if f > max_fmeasure:
            max_fmeasure = f
            max_epoch = i
            if args.save:
                model.save(tag2id, tokenizer=tokenizer, path=args.saved_path)
                print("\n")
                print('model saved with max_fmeasure {:.3%}'.format(max_fmeasure))
            else:
                print(f'get max_fmeasure {f}')
        print("max={:.3%} epoch={}".format(max_fmeasure, max_epoch))

