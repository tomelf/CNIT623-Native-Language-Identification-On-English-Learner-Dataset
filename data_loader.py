from conllu import parse, parse_tree

def load_raw_conllu():
    filepaths = [
        "dataset/UD_English-ESL/data/en_esl-ud-train.conllu", 
        "dataset/UD_English-ESL/data/corrected/en_cesl-ud-train.conllu", 
        "dataset/UD_English-ESL/data/en_esl-ud-dev.conllu", 
        "dataset/UD_English-ESL/data/corrected/en_cesl-ud-dev.conllu", 
        "dataset/UD_English-ESL/data/en_esl-ud-test.conllu", 
        "dataset/UD_English-ESL/data/corrected/en_cesl-ud-test.conllu"
    ]
    ret_list = []
    for path in filepaths:
        with open(path, "r") as f:
            raw_data = f.read()
        ret_list.append(parse(raw_data))
    return ret_list

def main():
    train_data, train_data_corrected, \
    dev_data, dev_data_corrected, \
    test_data, test_data_corrected = load_raw_conllu()
    
    print("# of train data: {}".format(len(train_data)))
    print("# of dev data: {}".format(len(dev_data)))
    print("# of test data: {}".format(len(test_data)))

if __name__ == "__main__":
    main()