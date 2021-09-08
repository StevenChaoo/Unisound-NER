# Author: StevenChaoo
# -*- coding:UTF-8 -*-


import argparse


def main(args):
    gold_file = open("./data/raw/test.txt", "r")
    pred_file = open("./{}".format(args.path), "r")
    composed_file = open("./composed.txt", "w")

    raw_text_list = []
    raw_text = ""
    for line in gold_file.readlines():
        line_list = line.strip().split(" ")
        if len(line_list) > 1:
            raw_text += line_list[0]
        else:
            raw_text_list.append(raw_text)
            raw_text = ""
    num = 0
    sent_num = 0
    for line in pred_file.readlines():
        if len(line.strip().split(" ")) > 1:
            new_line = "{} {}".format(raw_text_list[num][sent_num], line) 
            sent_num += 1
            composed_file.write(new_line)
        else:
            composed_file.write("\n")
            sent_num = 0
            num += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    args = parser.parse_args()
    main(args)
