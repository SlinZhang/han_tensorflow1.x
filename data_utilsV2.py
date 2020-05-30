import os
from collections import Counter
import config
import re
import numpy as np
import pickle

train_neg_base = "./data/train/neg/"
train_pos_base = "./data/train/pos"
test_neg_base = "./data/test/neg"
test_pos_base = "./data/test/pos"
base_paths = [train_neg_base, train_pos_base, test_neg_base, test_pos_base]


def read_data(base_paths):
    train_path_and_labels = {}
    test_path_and_labels = {}
    all_file = []
    all_data = []
    label_encode = {"neg": 0, "pos": 1}
    for base_path in base_paths:
        label = base_path.split("/")[3]
        label = label_encode[label]
        files = os.listdir(base_path)
        for file in files:
            path = os.path.join(base_path, file)
            if "train" in path:
                train_path_and_labels[path] = label
            else:
                test_path_and_labels[path] = label
            all_file.append(path)
    for file in all_file:
        with open(file, "r")as f:
            all_data.append(f.read())
    all_data = [data.replace("<br /><br />", "") for data in all_data]
    return all_data, train_path_and_labels, test_path_and_labels


def make_vocab(all_data):
    text = " ".join(all_data)
    counter_words = Counter(text.split(" "), )
    most_common_words = counter_words.most_common(1000)
    word2index = {"PAD": 0}
    for index, word in enumerate(most_common_words, start=1):
        word2index[word[0]] = index
    word2index["UNK"] = len(word2index)
    index2word = {index: word for word, index in word2index.items()}
    return word2index, index2word


def make_dataset(base_paths):
    sentence_split_symbol = ".|!"
    train_data = {"text": [], "label": [], "sentence_length": []}
    test_data = {"text": [], "label": [], "sentence_length": []}
    all_data, train_path_and_labels, test_path_and_labels = read_data(base_paths)
    word2index, index2word = make_vocab(all_data)
    for path, label in train_path_and_labels.items():
        temp_text = []
        temp_length = []
        with open(path, "r")as f:  # reda doc for train dataset
            content = f.read()
            sentences = content.split(".")
            sentences = sentences[:config.max_s]
            train_data["label"].append(label)
            for s in sentences:
                s = s.split(" ")[:config.max_w]
                if "" in s: s.remove("")
                if len(s) == 0:
                    continue
                word_index = []
                for word in s:
                    try:
                        word_index.append(word2index[word])
                    except:
                        word_index.append(word2index["UNK"])
                length = len(word_index)
                word_index = word_index + [0] * (config.max_w - len(word_index))
                temp_length.append(length)
                temp_text.append(word_index)
        if len(temp_text) < config.max_s:
            for _ in range(0, config.max_s - len(temp_text)):
                temp_text.append([0 for _ in range(0, config.max_w)])
                temp_length.append(0)
        train_data["text"].append(temp_text)
        train_data["sentence_length"].append(temp_length)


    with open("./data/train.pkl", "wb")as f:
        pickle.dump(train_data, f)

    for path, label in test_path_and_labels.items():
        temp_text = []
        temp_length = []
        with open(path, "r")as f:  # reda doc for test dataset
            content = f.read()
            sentences = content.split(".")
            sentences = sentences[:config.max_s]
            test_data["label"].append(label)
            for s in sentences:
                s = s.split(" ")[:config.max_w]
                if "" in s: s.remove("")
                if len(s) == 0:
                    continue
                word_index = []
                for word in s:
                    try:
                        word_index.append(word2index[word])
                    except:
                        word_index.append(word2index["UNK"])
                length = len(word_index)
                word_index = word_index + [0] * (config.max_w - len(word_index))
                temp_length.append(length)
                temp_text.append(word_index)
        if len(temp_text) < config.max_s:
            for _ in range(0, config.max_s - len(temp_text)):
                temp_text.append([0 for _ in range(0, config.max_w)])
                temp_length.append(0)
        test_data["text"].append(temp_text)
        test_data["sentence_length"].append(temp_length)

    with open("./data/test.pkl", "wb")as f:
        pickle.dump(test_data, f)


if __name__ == "__main__":
    train_neg_base = "./data/train/neg/"
    train_pos_base = "./data/train/pos"
    test_neg_base = "./data/test/neg"
    test_pos_base = "./data/test/pos"
    base_paths = [train_neg_base, train_pos_base, test_neg_base, test_pos_base]
    make_dataset(base_paths)
