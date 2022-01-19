import json
import os
import os.path as osp
import sys
import string
import copy
from natsort import natsorted
from tqdm import trange
from textblob import TextBlob



def remove_punct(s):
    _punctuation = set(string.punctuation)
    for punct in set(s).intersection(_punctuation):
        s = s.replace(punct, ' ')
    return ' '.join([f.lower() for f in s.split()])


def num_2word(s):
    swapdict = {'0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five',
                '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine', '10': 'ten'}
    ret = []
    for item in s:
        if str(item) in swapdict:
            ret.append(swapdict[str(item)])
        else:
            ret.append(item)
    return ret


def nvlr_scene_parser(scenes_path='data/', mode='val'):
    if mode == 'val': mode = 'dev'
    with open(scenes_path + f'/{mode}.json', 'r') as fin:
        file = '{\"scenes\":[' + ','.join([f.strip() for f in fin.readlines()]) + ']}'
        scenes = json.loads(file)
    return scenes


def nvlr_append_datasets(dataset_1, dataset_2, give_name='mixed', save=False):
    print(f"Dataset 1 size: {len(dataset_1['scenes'])}\n")
    print(f"Dataset 2 size: {len(dataset_2['scenes'])}\n")

    for scene in dataset_2['scenes']:
        dataset_1['scenes'].append(scene)

    print(f"Joint Dataset size: {len(dataset_1['scenes'])}\n")
    if save:
        with open('data/' + give_name + '.json', 'w') as fout:
            json.dump(dataset_1, fout)
    return dataset_1


def prepare_train_dev():
    train_json = nvlr_scene_parser(scenes_path='../data/', mode='train')
    dev_json = nvlr_scene_parser(scenes_path='../data/', mode='val')
    x = nvlr_append_datasets(train_json, dev_json, give_name='mixed', save=False)

    question_tokens = {}
    for scene in x['scenes']:
        corrected_sentence = str(TextBlob(scene['sentence']).correct())
        list_of_words = num_2word(remove_punct(corrected_sentence.strip()).split(' '))
        for f in list_of_words:
            if f in question_tokens:
                question_tokens[f] += 1
            else:
                question_tokens.update({f: 1})

    print("Saving Dictionary With Frequencies...\n")
    with open('../question_vocabulary_clean.json', 'w') as fout:
        json.dump(question_tokens, fout)


def reduce_and_resave_traindev(vocabulary_path='question_vocabulary_clean.json', cutoff_threshold=10):
    print(f"Opening  data...\n")
    train_json = nvlr_scene_parser(scenes_path='data/', mode='train')
    dev_json = nvlr_scene_parser(scenes_path='data/', mode='val')
    data = nvlr_append_datasets(train_json, dev_json, give_name='mixed', save=False)
    print(f"Opening vocabulary...\n")
    try:
        with open(vocabulary_path, 'r') as fin:
            vocab = json.load(fin)
    except FileNotFoundError:
        print("File not found... lets generate it!\n")
        prepare_train_dev()
        with open(vocabulary_path, 'r') as fin:
            vocab = json.load(fin)

    clean = {'scenes': []}
    for scene in data['scenes']:
        # Clean the sentence #
        corrected_sentence = str(TextBlob(scene['sentence']).correct())
        list_of_words = num_2word(remove_punct(corrected_sentence.strip()).split(' '))
        flag = True
        for f in list_of_words:
            if f in vocab:
                if vocab[f] < cutoff_threshold:
                    flag = False
                    break
            else:
                raise ValueError(f"How come {f} is not in the vocabulary?\n")

        if flag:
            new_scene = copy.deepcopy(scene)
            new_scene['sentence'] = ' '.join(list_of_words)
            clean['scenes'].append(new_scene)
        else:
            pass
    with open('clean_data/clean_traindev.json', 'w') as fout:
        json.dump(clean, fout)
    return


def reduce_and_resave_test(vocabulary_path='question_vocabulary_clean.json', cutoff_threshold=10):
    print(f"Opening  data...\n")
    data = nvlr_scene_parser(scenes_path='data/', mode='test')
    print(f"Opening vocabulary...\n")
    try:
        with open(vocabulary_path, 'r') as fin:
            vocab = json.load(fin)
    except FileNotFoundError:
        print("File not found... lets generate it!\n")
        prepare_train_dev()
        with open(vocabulary_path, 'r') as fin:
            vocab = json.load(fin)

    clean = {'scenes': []}
    for scene in data['scenes']:
        # Clean the sentence #
        corrected_sentence = str(TextBlob(scene['sentence']).correct())
        list_of_words = num_2word(remove_punct(corrected_sentence.strip()).split(' '))
        flag = True
        for f in list_of_words:
            if f in vocab:
                if vocab[f] < cutoff_threshold:
                    flag = False
                    print(
                        f"The word {f} of sentence {' '.join(list_of_words)} is below threshold {cutoff_threshold}!\n")
                    break
            else:
                # raise ValueError(f"How come {f} is not in the vocabulary?\n")
                # print(f"How come {f} is not in the vocabulary?\n")
                flag = False
                pass
                # Consecutive #
                # Close #

        if flag:
            new_scene = copy.deepcopy(scene)
            new_scene['sentence'] = ' '.join(list_of_words)
            clean['scenes'].append(new_scene)
        else:
            pass
    with open('clean_data/clean_test.json', 'w') as fout:
        json.dump(clean, fout)
    return


def create_translation_dict(vocabulary_path='../question_vocabulary_clean.json'):
    print(f"Opening vocabulary...\n")
    try:
        with open(vocabulary_path, 'r') as fin:
            vocab = json.load(fin)
    except FileNotFoundError:
        print("File not found... lets generate it!\n")
        prepare_train_dev()
        with open(vocabulary_path, 'r') as fin:
            vocab = json.load(fin)

    print("Lets create the translation dict\n")
    vocab = natsorted(vocab)
    translation = {'NULL':0, '<START>': 1, '<END>': 2}
    for index, item in enumerate(vocab):
        translation.update({item: index + 2})

    print("Lets save it!\n")
    with open('../translation_vocabulary.json', 'w') as fout:
        json.dump(translation, fout)

    return


def nvlr_translation_parser():
    try:
        with open('../translation_vocabulary.json', 'r') as fin:
            data = json.load(fin)
    except FileNotFoundError:
        create_translation_dict()
        with open('../translation_vocabulary.json', 'r') as fin:
            data = json.load(fin)
    return data
