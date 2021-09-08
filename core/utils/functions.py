import torch
from .alphabet import Alphabet
import re
import pdb


def extract_tags_from_dataset(dp):
    tags = set()
    with open(dp) as f:
        for i, line in enumerate(f):
            line = line.rstrip('\n')
            if line == '':
                continue

            # Get tag
            if line.lstrip() != line:
                tag = line.lstrip()
            else:
                tag = line.split(" ")[1]

            # No "S-" tag in training data
            if tag.split('-').__len__() == 2:
                tag = tag.split('-')[1]
            tags.add(tag)
    for tag in tags.copy():
        if tag == 'O':
            continue
        tags.remove(tag)
        for prefix in ['B', 'I', 'E']:
            tags.add(prefix + '-' + tag)
    for tag in ['[CLS]', '[SEP]', '[PAD]']:
        tags.add(tag)
    return tags


def strQ2B(string):
    for q, b in Q2Bdic.items():
        string = string.replace(q, b)
    return string


class Batch():
    def __init__(self, input_ids, input_tags, pad=0, device='gpu'):
        self.input_ids = input_ids.to(device)
        if input_tags is not None:
            self.input_tags = input_tags.to(device)
        self.mask = (input_ids != pad).to(device)
        self.lengths = self.mask.sum(dim=-1)

    def to(self, device):
        self.input_ids = self.input_ids.to(device)
        if self.input_tags is not None:
            self.input_tags = self.input_tags.to(device)
        self.mask = self.mask.to(device)


def entity_match(epoch, batch_target_labels, batch_predict_labels, batch_mask, id2tag):
    """
    batch_target_labels: torch.tensor
    return: total_target_ners, total_predict_ners, total_correct_ners
    """
    target_labels_list = [[id2tag[id_.item()] for id_ in l] 
                          for l in batch_target_labels]
    predict_labels_list = [[id2tag[id_.item()] for id_ in l]
                           for l in batch_predict_labels]
    batch_mask = batch_mask.tolist()
    total_target_ners = []
    total_predict_ners = []
    total_correct_ners = []
    pred_file = open("/home/ZhengyiZhao/File/BERTCRF/pred/pred_file{}.txt".format(epoch), "a")
    for target_labels, predict_labels, masks in zip(target_labels_list, predict_labels_list, batch_mask):
        target_ners = extract_ner(target_labels, masks)
        predict_ners = extract_ner(predict_labels, masks)
        correct_ners = list(set(target_ners).intersection(set(predict_ners)))
        for i in range(len(target_labels)):
            if target_labels[i] == "[CLS]":
                continue
            elif target_labels[i] == "[SEP]":
                break
            else:
                label_pair = "{} {}".format(target_labels[i], predict_labels[i])
                pred_file.write(label_pair)
                pred_file.write("\n")
        pred_file.write("\n")
        total_target_ners += target_ners
        total_predict_ners += predict_ners
        total_correct_ners += correct_ners
    precision = (len(total_correct_ners) + 1) / (len(total_predict_ners) + 1)
    recall = (len(total_correct_ners) + 1) / (len(total_target_ners) + 1)
    f_measure = 2 * precision * recall / (precision + recall)
    return total_target_ners, total_predict_ners, total_correct_ners


def test(model, dataset_loader, id2tag, device='cpu'):
    total_target_ners = []
    total_predict_ners = []
    total_correct_ners = []
    for i, batch in enumerate(dataset_loader.traversal(device)):
        with torch.no_grad():
            predict_labels = model.decode(batch.input_ids, batch.mask)
            max_seq_len = max([len(seq) for seq in predict_labels])
            predict_labels = [seq + [0] * (max_seq_len - len(seq)) for seq in predict_labels]
            predict_labels = torch.tensor(predict_labels)
            target_ners, predict_ners, correct_ners = entity_match(batch.input_tags, predict_labels, batch.mask, id2tag)
            total_target_ners.extend(target_ners)
            total_predict_ners.extend(predict_ners)
            total_correct_ners.extend(correct_ners)
    precision = (len(total_correct_ners) + 1) / (len(total_predict_ners) + 1)
    recall = (len(total_correct_ners) + 1) / (len(total_target_ners) + 1)
    f_measure = 2 * precision * recall / (precision + recall)
    print(f'precision={precision} recall={recall} f_measure={f_measure}')
    label_evaluate(total_target_ners, total_predict_ners, total_correct_ners)
    return precision, recall, f_measure
    # logging.info(str(precision) + '\t' + str(recall) + '\t' + str(f_measure))


def label_evaluate(target_ners, predict_ners, correct_ners):
    label_info = {}
    for entity in target_ners:
        label = re.search('(.*?)\[', entity).group(1)
        if label not in label_info:
            label_info.update({label:{'correct': 0, 'predict': 0, 'target': 0}})
        label_info[label]['target'] += 1
    for entity in predict_ners:
        label = re.search('(.*?)\[', entity).group(1)
        if label not in label_info:
            label_info.update({label:{'correct': 0, 'predict': 0, 'target': 0}})
        label_info[label]['predict'] += 1
    for entity in correct_ners:
        label = re.search('(.*?)\[', entity).group(1)
        if label not in label_info:
            label_info.update({label:{'correct': 0, 'predict': 0, 'target': 0}})
        label_info[label]['correct'] += 1
    for label in label_info:
        precision = (label_info[label]['correct'] + 1) / (label_info[label]['predict'] + 1)
        recall = (label_info[label]['correct'] + 1) / (label_info[label]['target'] + 1)
        f_measure = 2 * precision * recall / (precision + recall)
        print(label)
        print('p={:.3%} r={:.3%} f={:.3%}'.format(precision, recall, f_measure))

                          
def extract_ner(labels, masks=None):
    """
    labels: list
    return: entities list
    根据labels提取一句话中的实体, 并输出实体list
    """
    if masks is None:
        masks = [1] * len(labels)
    entities = []
    cur_entity = ""
    for i, (label, mask) in enumerate(zip(labels, masks)):
        if mask == 0:
            break
        if (label == "[CLS]") or (label == "[SEP]"):
            cur_entity = ""
        elif label.startswith("B-"):
            label = label[2:]
            cur_entity = label + "[" + str(i)
        elif label.startswith("I-"):
            label = label[2:]
            if not cur_entity.startswith(label):
                cur_entity = ""
        elif label.startswith("M-"):
            label = label[2:]
            if not cur_entity.startswith(label):
                cur_entity = ""
        elif label.startswith("E-"):
            label = label[2:]
            if cur_entity.startswith(label):
                cur_entity = cur_entity + '-' + str(i) + ']'
                entities.append(cur_entity)
            cur_entity = ""
        elif label.startswith("O"):
            cur_entity = ""
        elif label.startswith("S-"):
            label = label[2:]
            cur_entity = label + "[" + str(i) + '-' + str(i) + "]"
            entities.append(cur_entity)
            cur_entity = ""
        elif label == "[PAD]":
            cur_entity = ""
        else:
            print(label, mask)
            raise Exception
    return entities


if __name__ == "__main__":
    alphabet = Alphabet(name='char_alphabet', keep_growing=False)
    alphabet.load(input_directory='../data/embeddings')
    data_path = '../data/cdd/data.txt'
    for chars, tags in iter_sample(data_path):
        for char in chars:
            id_ = alphabet.get_index(char)
            if id_ == 1: print(id_)
