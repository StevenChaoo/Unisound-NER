# Author: StevenChaoo
# -*- coding:utf-8 -*-


import argparse


file_path = "./BIEO.txt"


def getEntity(label_type, file_path):
    f = open(file_path, "r")
    labels = []
    label_map = {"gold": 1, "pred": 2}
    for idx, line in enumerate(f.readlines()):
        label_list = line.strip().split(" ")
        if len(label_list) > 1:
            if label_list[label_map[label_type]][0] == "B":
                label = label_list[label_map[label_type]][2:]
                start_pos = idx
            if label_list[label_map[label_type]][0] == "E":
                end_pos = idx
                entity = [label, (start_pos, end_pos)]
                labels.append(entity)
    f.close()
    return labels


def countLabel(labels):
    label_map = {"DISEASE": 0, "PURPOSE": 0, "PERSON_GROUP": 0, "CONDITAION": 0, "SYMPTOM": 0, "INAPPLICABILITY": 0}
    for gl in labels:
        label_map[gl[0]] += 1
    return label_map
    

def countRelLabel(gold_labels, pred_labels):
    label_map = {"DISEASE": 0, "PURPOSE": 0, "PERSON_GROUP": 0, "CONDITAION": 0, "SYMPTOM": 0, "INAPPLICABILITY": 0}
    for gl in gold_labels:
        for pl in pred_labels:
            if gl == pl:
                label_map[gl[0]] += 1
    return label_map


def main():
    gold_labels = getEntity("gold", file_path)
    pred_labels = getEntity("pred", file_path)

    gold_map = countLabel(gold_labels)
    pred_map = countLabel(pred_labels)
    rel_map = countRelLabel(gold_labels, pred_labels)

    for label in rel_map.keys():
        print(label)
        P = rel_map[label]/pred_map[label]
        R = rel_map[label]/gold_map[label]
        F = (2*P*R)/(P+R)
        print("p={:.3%}  r={:.3%}  f={:.3%}".format(P, R, F))


if __name__ == "__main__":
    main()
