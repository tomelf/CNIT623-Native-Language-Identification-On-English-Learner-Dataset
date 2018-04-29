import conllu
import os
import pandas as pd
import numpy as np
import re

from os import listdir
from os.path import isfile, isdir, join
from xml.etree import ElementTree

def conllu_meta_parse_line(line):
    id_m = re.match(r"#ID=\/[0-9_]+\/([\w]+).xml", line)
    sent_m = re.match(r"#SENT=([^\n]+)", line)
    if id_m:
        return id_m.group(1)
    if sent_m:
        return sent_m.group(1)

def conllu_meta_parse(text):
    ns_types_df = pd.read_csv("dataset/UD_English-ESL/ns_type.csv")
    ns_types_list = ns_types_df["NS_TYPE"].tolist()
    meta_infos = [
        [
            conllu_meta_parse_line(line)
            for line in sentence.split("\n")
            if line and line.strip().startswith("#")
        ]
        for sentence in text.split("\n\n")
        if sentence
    ]
    ret_list = []
    for meta_info in meta_infos:
        error_list = re.findall('<ns type=\"([^\"]+)\">', meta_info[1], flags=re.IGNORECASE)
        # errors = dict((ns, error_list.count(ns)) for ns in set(error_list))
        errors = error_list
        ret_list.append({"doc_id": meta_info[0], "sent": meta_info[1], "errors": errors})
    return ret_list

def load_raw_conllu(load_train=True, load_dev=True, load_test=True):
    print("Load raw conllu dataset (load_train={}, load_dev={}, load_test={})".format(load_train, load_dev, load_test))
    filepaths = []
    if load_train:
        filepaths += [
            "dataset/UD_English-ESL/data/en_esl-ud-train.conllu", 
            "dataset/UD_English-ESL/data/corrected/en_cesl-ud-train.conllu",
        ]
    if load_dev:
        filepaths += [
            "dataset/UD_English-ESL/data/en_esl-ud-dev.conllu", 
            "dataset/UD_English-ESL/data/corrected/en_cesl-ud-dev.conllu", 
        ]
    if load_test:
        filepaths += [
            "dataset/UD_English-ESL/data/en_esl-ud-test.conllu", 
            "dataset/UD_English-ESL/data/corrected/en_cesl-ud-test.conllu",
        ]
    data_list = []
    meta_list = []
    for path in filepaths:
        print("Processing {}".format(path))
        with open(path, "r") as f:
            raw_data = f.read()
        data_list.append(conllu.parse(raw_data))
        meta_list.append(conllu_meta_parse(raw_data))
        
    return meta_list, data_list

def load_post_metadata(with_ans=False):
    filepath = "dataset/UD_English-ESL/fce-released-dataset/dataset/"
    subpaths = [f for f in listdir(filepath) if isdir(join(filepath, f))]
    stats_raw = []
    
    for subpath in subpaths:
        subpath = join(filepath, subpath)
        files = [f for f in listdir(subpath) if isfile(join(subpath, f))]
        for file in files:
            l_id = file.split('.')[0]
            file = join(subpath, file)
            
            l_native_lan = None
            l_age = None
            l_score = None
            l_ans1 = None
            l_ans1_errors = []
            l_ans1_score = None
            l_ans2 = None
            l_ans2_errors = []
            l_ans2_score = None
            
            learner = ElementTree.parse(file).getroot()
            head = learner.find('head')
            candidate = head.find('candidate')
            personnel = candidate.find('personnel')
            if personnel.find('language') != None:
                l_native_lan = personnel.find('language').text
            if personnel.find('age') != None:
                l_age = personnel.find('age').text
            if candidate.find('score') != None:
                l_score = float(candidate.find('score').text)

            text = head.find('text')
            ans1 = text.find('answer1')
            if ans1 != None:
                if ans1.find('exam_score') != None:
                    l_ans1_score = ans1.find('exam_score').text
                if ans1.find('coded_answer') != None:
                    l_ans1 = ElementTree.tostring(ans1.find('coded_answer'), method='html').decode().strip()
                    l_ans1 = re.sub('<coded_answer>\n', '', l_ans1, flags=re.IGNORECASE).strip()
                    l_ans1 = re.sub('</coded_answer>', '', l_ans1, flags=re.IGNORECASE).strip()
                    l_ans1_errors = re.findall('<ns type=\"([^\"]+)\">', l_ans1, flags=re.IGNORECASE)
            ans2 = text.find('answer2')
            if ans2 != None:
                if ans2.find('exam_score') != None:
                    l_ans2_score = ans2.find('exam_score').text
                if ans2.find('coded_answer') != None:
                    l_ans2 = ElementTree.tostring(ans2.find('coded_answer'), method='html').decode().strip()
                    l_ans2 = re.sub('<coded_answer>\n', '', l_ans2, flags=re.IGNORECASE).strip()
                    l_ans2 = re.sub('</coded_answer>', '', l_ans2, flags=re.IGNORECASE).strip()
                    l_ans2_errors = re.findall('<ns type=\"([^\"]+)\">', l_ans2, flags=re.IGNORECASE)

            row = [l_id, l_native_lan, l_age, l_score, 
                   l_ans1, l_ans1_errors, l_ans1_score, 
                   l_ans2, l_ans2_errors, l_ans2_score]
            stats_raw.append(row)
    cols = ['doc_id', 'native_language', 'age_range', 'score', 
            "ans1", "ans1_errors", "ans1_score", 
            "ans2", "ans2_errors", "ans2_score"]
    stats_df = pd.DataFrame(stats_raw, columns=cols)
    return stats_df

def load_data(load_train=True, load_dev=True, load_test=True):
    print("Load data (load_train={}, load_dev={}, load_test={})".format(load_train, load_dev, load_test))
    raw_meta_list, raw_data_list = load_raw_conllu(load_train, load_dev, load_test)
    print("Build metadata")
    meta_list = []
    for meta in raw_meta_list:
        cols = list(meta[0].keys())
        metas = []
        for i in range(len(meta)):
            values = list(meta[i].values())
            metas.append(values)
        meta_list.append(pd.DataFrame(metas, columns=cols))
        meta_list[-1].insert(0, 'id', range(1, len(meta_list[-1])+1))
    print("Build sentences")
    data_list = []
    for idx, data in enumerate(raw_data_list):
        meta = meta_list[idx]
        sentence_dfs = []
        cols = list(data[0][0].keys()) + ["meta_id"]
        for i in range(len(data)):
            words = []
            for j in range(len(data[i])):
                words.append(list(data[i][j].values()) + [meta["id"][i]])
            sentence_dfs.append(pd.DataFrame(words, columns=cols).set_index('id'))
        data_list.append(sentence_dfs)
    post_df = load_post_metadata()
    for i in range(len(meta_list)):
        meta_list[i] = meta_list[i].set_index('doc_id').join(post_df.set_index('doc_id'))
        meta_list[i] = meta_list[i].reset_index().set_index('id').sort_index()
    print("=================")
    return meta_list, data_list

def dump_preprocessed_data(name, meta, data, format="csv"):
    print("Dump {} to preprocessed/{}".format(name, name))
    directory = "preprocessed/{}".format(name)
    if not os.path.exists(directory):
        os.makedirs(directory)  
    if format == "csv":
        meta.to_csv("{}/meta.csv".format(directory))
    else:
        meta.to_json("{}/meta.json".format(directory))
    for d in data:
        if format == "csv":
            d.to_csv("{}/{}.csv".format(directory, d["meta_id"].unique()[0]))
        else:
            d.to_json("{}/{}.json".format(directory, d["meta_id"].unique()[0]))

def main():
    meta_list, data_list = load_data(load_train=True, load_dev=True, load_test=True)

    train_meta, train_meta_corrected, \
    dev_meta, dev_meta_corrected, \
    test_meta, test_meta_corrected = meta_list
    
    train_data, train_data_corrected, \
    dev_data, dev_data_corrected, \
    test_data, test_data_corrected = data_list
    
    print(train_meta.iloc[0].tolist())
    
    # f = "csv" # or "json"
    # dump_preprocessed_data("train", train_meta, train_data, format=f)
    # dump_preprocessed_data("train_corrected", train_meta_corrected, train_data_corrected, format=f)
    # dump_preprocessed_data("dev", dev_meta, dev_data, format=f)
    # dump_preprocessed_data("dev_corrected", dev_meta_corrected, dev_data_corrected, format=f)
    # dump_preprocessed_data("test", test_meta, test_data, format=f)
    # dump_preprocessed_data("test_corrected", test_meta_corrected, test_data_corrected, format=f)

if __name__ == "__main__":
    main()