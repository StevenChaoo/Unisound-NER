from torch.utils.data import Sampler, Dataset, DataLoader
from .functions import Batch
from random import randint
from tqdm import tqdm
import random
import torch
import pdb
from utils import process


class NERSampler(Sampler):
    def __init__(self, data_source, visible_tags=None, max_len=None, split_method=None):
        super(NERSampler).__init__()
        self.data_source = data_source
        self.visible_tags = visible_tags
        self.max_len = max_len
        self.split_method = split_method

    def __iter__(self):
        with open(self.data_source) as f:
            lines = f.readlines()
        chars, labels = ['[CLS]'], ['[CLS]']
        for line in lines:
            line = line.strip('\n')
            if line == '':
                chars.append('[SEP]')
                labels.append('[SEP]')
                if self.split_method == 'sentence': 
                    for chars_short, tags_short in self.split_by_sentence(chars[1: -2], labels[1: -2]):
                        yield chars_short, tags_short
                else:
                    yield chars, labels
                chars, labels = ['[CLS]'], ['[CLS]']
            elif line.startswith(' '):
                char = " "
                label = line.strip()
                if self.visible_tags is not None:
                    if label not in self.visible_tags:
                        label = 'O'
                chars.append(char)
                labels.append(label)
            else:
                try:
                    char, label = line.split(' ')[0: 2]
                except ValueError:
                    raise Exception(line)
                if self.visible_tags is not None:
                    if label not in self.visible_tags:
                        label = 'O'
                chars.append(char)
                labels.append(label)

    def split_by_lenth(self, chars, tags):
        separators = ['。', '！', '》']
        cur_chars, cur_tags = [], []
        for char, tag in zip(chars, tags):
            cur_chars.append(char)
            cur_tags.append(tag)
            if len(cur_chars) == self.max_len:
                yield cur_chars, cur_tags
                cur_chars, cur_tags = [], []
        yield cur_chars, cur_tags

    def split_by_sentence(self, chars, tags):
        separators = ['。', '！', '？']
        cur_chars, cur_tags = ['[CLS]'], ['[CLS]']
        for char, tag in zip(chars, tags):
            cur_chars.append(char)
            cur_tags.append(tag)
            if char in separators:
                cur_chars.append('[SEP]')
                cur_tags.append('[SEP]')
                yield cur_chars, cur_tags
                cur_chars, cur_tags = ['[CLS]'], ['[CLS]']
        cur_chars.append('[SEP]')
        cur_tags.append('[SEP]')
        yield cur_chars, cur_tags


class NERSampler_nolabel(Sampler):
    def __init__(self, data_source):
        super(NERSampler_nolabel).__init__()
        self.data_source = data_source

    def __iter__(self):
        lines = open(self.data_source).readlines()
        sent = ['[CLS]']
        for line in lines:
            line = line.strip('\n')
            if line != '':
                char = line
                sent.append(char)
            else:
                sent.append('[SEP]')
                tags = len(sent) * ['O']
                yield sent, tags
                sent = ['[CLS]']


class NERDataset(Dataset):
    def __init__(self, data_source=None, nolabel=False, visible_tags=None, hidden_tags=None, ratio=1.0):
        super(NERDataset).__init__()
        self.text_list = []
        self.tags_list = []
        self.ratio = ratio
        self.visible_tags = visible_tags
        self.hidden_tags = hidden_tags
        if data_source is not None:
            self.read(data_source, nolabel)
        print(f'dataset read over from {data_source}')

    def __getitem__(self, index):
        return self.text_list[index], self.tags_list[index]

    def __len__(self):
        return len(self.text_list)

    def __add__(self, x):
        self.text_list += x.text_list
        self.tags_list += x.tags_list
        return self

    def read(self, data_source, nolabel):
        if nolabel:
            sampler = NERSampler_nolabel(data_source)
        else:
            sampler = NERSampler(data_source, self.visible_tags)
        for text, tags in sampler:
            if len(text) > 512: print(text)
            if self.hidden_tags is not None:
                text, tags = process.hide_tags(text, tags, self.hidden_tags)
            if random.random() < self.ratio:
                self.text_list.append(text)
                self.tags_list.append(tags)

    def split(self, ratio=None, fold=None, seed=None):
        if seed is not None:
            random.seed(seed)
        trainset = NERDataset()
        testset = NERDataset()
        # get random indexes
        if fold is not None:
            ratio = 1 - 1 / fold
        else:
            fold = 1
        size = (1 - ratio) * len(self)  # size of testset
        indexes = random.sample(x, len(self.text_list))
        for i in range(fold):
            for i, (text, tags) in enumerate(zip(self.text_list, self.tags_list)):
                if fold * size < i < (fold + 1) * size:
                    testset.text_list.append(text)
                    testset.tags_list.append(tags)
                else:
                    trainset.text_list.append(text)
                    trainset.tags_list.append(tags)
            yield trainset, testset


class NERDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, alphabet, tag_id):
        super(NERDataLoader).__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.alphabet = alphabet
        self.tag_id = tag_id
        self.indexes = list(range(len(dataset)))

    def __iter__(self):
        while True:
            random.shuffle(self.indexes)
            batch_text, batch_tags = [], []
            for i, index in enumerate(self.indexes):
                text, tags = self.dataset[index]
                batch_text.append(text)
                batch_tags.append(tags)
                if i % self.batch_size == 0:
                    yield self.collate_fn(batch_text, batch_tags)
                    batch_text, batch_tags = [], []

    def traversal(self, device='cpu'):
        batch_text, batch_tags = [], []
        zip_list = []
        for i in range(len(self.dataset.text_list)):
            zip_list.append((self.dataset.text_list[i], self.dataset.tags_list[i]))
        for i, (text, tags) in enumerate(tqdm(zip_list)):
            batch_text.append(text)
            batch_tags.append(tags)
            if (i + 1) % self.batch_size == 0:
                yield self.collate_fn(batch_text, batch_tags, device)
                batch_text, batch_tags = [], []
        if len(batch_text) == 0: return
        yield self.collate_fn(batch_text, batch_tags, device)

    def collate_fn(self, batch_text, batch_tags, device='cpu'):
        alphabet = self.alphabet
        label_to_id_dic = self.tag_id
        max_len = max(len(text) for text in batch_text)
        for text, tags in zip(batch_text, batch_tags):
            text += ["[PAD]"] * (max_len - len(text))
            tags += ["[PAD]"] * (max_len - len(tags))
        batch_textId = [alphabet.convert_tokens_to_ids(text) for text in batch_text]
        batch_tagsId = [[label_to_id_dic[tag] for tag in tags] for tags in batch_tags]
        batch_text_tensor = torch.LongTensor(batch_textId)
        batch_tags_tensor = torch.LongTensor(batch_tagsId)
        batch_text, batch_tags = [], []
        return Batch(batch_text_tensor, batch_tags_tensor, device=device)
