# make csv files into huggingface datasets formant and make vocabulary

import ast
import json
import os
import re

import evaluate
import IPython.display as ipd
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tgt
from datasets import (
    Audio,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
    load_metric,
)

SPEECH_OCEAN_PATH = "/data2/haeyoung/speechocean762/"


# make SPEECHOCEAN762 utterance list to huggingface datasets
# 'real' refers to non-native realized phone sequences that include SID. Used for forced-alignment
train_ocean_ds = load_dataset("csv", data_files="./speechocean_train.csv", delimiter="|", split="train")
test_ocean_ds = load_dataset("csv", data_files="./speechocean_test.csv", delimiter="|", split="train")
ocean_phone_set = set()

w_stress = set()
w_acc = set()
w_tot = set()
p_acc = set()

tot = set()
comp = set()
pros = set()
flu = set()
acc = set()


def load_ocean(batch):
    batch["completeness"] = int(round(batch["completeness"]))

    batch["w_total"] = ast.literal_eval(batch["w_total"])
    batch["w_accuracy"] = ast.literal_eval(batch["w_accuracy"])
    batch["w_stress"] = ast.literal_eval(batch["w_stress"])
    batch["p_accuracy"] = ast.literal_eval(batch["p_accuracy"])
    batch["phone"] = ast.literal_eval(batch["phone"])
    batch["canon"] = ast.literal_eval(batch["canon"])
    batch["real"] = ast.literal_eval(batch["real"])
    batch["mispronunciations"] = ast.literal_eval(batch["mispronunciations"])

    batch["path"] = SPEECH_OCEAN_PATH + batch["path"]
    for i in range(len(batch["p_accuracy"])):
        batch["p_accuracy"][i] = int(batch["p_accuracy"][i] * 5)

    w_tot.update(batch["w_total"])
    w_acc.update(batch["w_accuracy"])
    w_stress.update(batch["w_stress"])
    p_acc.update(batch["p_accuracy"])

    tot.add(batch["total"])
    comp.add(batch["completeness"])
    pros.add(batch["prosodic"])
    flu.add(batch["fluency"])
    acc.add(batch["accuracy"])

    return batch


def preprocess_ocean_phones(batch):
    canon_list = []
    real_list = []
    ans_list = []

    for i in range(len(batch["canon"])):
        canon_list.append(re.sub("[0-9*]", "", batch["canon"][i]))
        ans = batch["real"][i]
        ans = re.sub("[0-9*]", "", ans)
        ans = re.sub("<unk>", "ERR", ans)
        ans = re.sub("<DEL>", "", ans)

        real = batch["real"][i]
        real = re.sub("[0-9*]", "", real)
        real = re.sub("<unk>", "ERR", real)

        real_list.append(real)
        ans_list.append(ans)

    batch["canon"] = re.sub(" +", " ", " ".join(canon_list))
    batch["real"] = re.sub(" +", " ", " ".join(real_list))
    batch["ans"] = re.sub(" +", " ", " ".join(ans_list))
    if batch["ans"].strip() == "":
        batch["ans"] = "ERR"

    ocean_phone_set.update(batch["canon"].split())
    ocean_phone_set.update(batch["ans"].split())

    return batch


train_ocean_ds = train_ocean_ds.map(load_ocean)
train_ocean_ds = train_ocean_ds.map(lambda x: {"audio": x["path"]})
train_ocean_ds = train_ocean_ds.cast_column("audio", Audio(sampling_rate=16000))
train_ocean_ds = train_ocean_ds.map(preprocess_ocean_phones)

test_ocean_ds = test_ocean_ds.map(load_ocean)
test_ocean_ds = test_ocean_ds.map(lambda x: {"audio": x["path"]})
test_ocean_ds = test_ocean_ds.cast_column("audio", Audio(sampling_rate=16000))
test_ocean_ds = test_ocean_ds.map(preprocess_ocean_phones)

train_ocean_ds.save_to_disk("/data2/haeyoung/speechocean762/preprocess/speechocean_train_ds")
test_ocean_ds.save_to_disk("/data2/haeyoung/speechocean762/preprocess/speechocean_test_ds")


# make vocabulary set
def extract_all_chars_phone(batch):
    all_text = " ".join(batch["ans"])
    vocab = list(set(all_text.split()))
    vocab.append(" ")
    return {"vocab": [vocab], "all_text": [all_text]}


train_ds = load_from_disk("/data2/haeyoung/speechocean762/preprocess/speechocean_train_ds/")

train_vocab = train_ds.map(extract_all_chars_phone, batched=True, batch_size=-1, remove_columns=train_ds.column_names)

vocab_list = list(set(train_vocab["vocab"][0]))
vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}
vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)
len(vocab_dict)

with open("../vocab/vocab.json", "w") as vocab_file:
    json.dump(vocab_dict, vocab_file, ensure_ascii=False)

print("- Finished making datasets and vocabulary")
