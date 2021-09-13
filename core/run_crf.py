# Author: StevenChaoo
# -*- coding:UTF-8 -*-


import json
import logging
import time
import random
import sys

from sklearn_crfsuite import CRF
from sklearn.metrics import classification_report
from util import tools
from tqdm import tqdm


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger("root")


def sentence2feature(sentences):
    # Extract context features
    features = []
    for sentence in tqdm(sentences):
        result = []
        for i in range(len(sentence)):
            # Previous and next word
            word = sentence[i]
            previous_word = '<start>' if i == 0 else sentence[i-1]
            next_word = '<end>' if i == (len(sentence)-1) else sentence[i+1]

            # Contains five features
            feature = {
                "w": word,
                "w-1": previous_word,
                "w+1": next_word,
                "w-1:w": previous_word+word,
                "w:w+1": word+next_word
            }
            result.append(feature)
        features.append(result)

    # Return results
    return features


def normalizationLabel(label_lists):
    labels = []
    for label_list in label_lists:
        for label in label_list:
            if len(label) > 1:
                labels.append(label[2:])
            else:
                labels.append(label)
    return labels


class CRFModel(object):
    def __init__(self):
        self.model = CRF(algorithm='l2sgd',
                         c2=0.1,
                         max_iterations=100)

    def train(self, features, tag_lists):
        self.model.fit(features, tag_lists)

    def evaluate(self, features, tag_lists):
        predict_tag = self.model.predict(features)
        real_tag = normalizationLabel(tag_lists)
        pred_tag = normalizationLabel(predict_tag)
        print(classification_report(real_tag, pred_tag))


def dataProcess(path):
    f = open(path, "r")
    word_lists = []
    label_lists = []
    word_list = []
    label_list = []
    for line in f.readlines():
        line_list = line.strip().split(" ")
        if len(line_list) > 1:
            word_list.append(line_list[0])
            label_list.append(line_list[1])
        else:
            word_lists.append(word_list)
            label_lists.append(label_list)
    return word_lists, label_lists


def main():
    # Prepare dataset
    train_word_lists, train_label_lists = dataProcess("./data/dis/train.txt")
    test_word_lists, test_label_lists = dataProcess("./data/dis/test.txt")

    # Extract features
    logger.info("Prepare train data")
    train_features = sentence2feature(train_word_lists)
    logger.info("Prepare test data")
    test_features = sentence2feature(test_word_lists)

    # Build CRF model
    logger.info("Build CRF model")
    crf = CRFModel()
    logger.info("Success!")

    # Train model
    logger.info("Begin training")
    crf.train(train_features, train_label_lists)
    logger.info("Finish training")

    # Evaluate model
    logger.info("Begin evaluating")
    crf.evaluate(test_features, test_label_lists)
    logger.info("Finish evaluating")


if __name__ == "__main__":
    # Main function
    main()
