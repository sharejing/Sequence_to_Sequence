import random
import re
import unicodedata

import torch
import numpy as np
from torch.autograd import Variable
from keras.preprocessing import sequence

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 10


class Lang(object):

    def __init__(self, name):

        self.name = name
        self.word2index = {}                       # 将词变为索引
        self.index2word = {0: "SOS", 1: "EOS"}     # 每一个索引对应的词
        self.word2count = {}                       # 统计每一个词的词频
        self.n_words = 2                           # 索引从 2 开始

    def addWord(self, word):

        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def addSentence(self, sentence):

        for word in sentence.split(' '):
            self.addWord(word)




def readLangs(lang1, lang2, reverse=False):

    print("Reading lines...")

    # 阅读文件，每行作为一个元素存入List中
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').read().strip().split('\n')

    # 将list中的每个元素以\t分割为源语言与目标语言，并将二者归一化
    # 形式如：[['源语言', '目标语言'], ['源语言', '目标语言']......]
    pairs = [[s for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances(源语言与目标语言之间可以相互转化)
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs




# 因为翻译对很多，数据集很大，所以这里对数据做一些处理。
#    1. 源语言与目标语言的词数不超过10个
#    2.目标语言以都是以eng_prefixes开头的
# p是一个list, 该list的第一个元素是源语言，第二个元素是目标语言
def filterPair(p):

    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH


# pairs是一个list, 但list中的每一个元素都是list
def filterPairs(pairs):

    return [pair for pair in pairs if filterPair(pair)]

'''
    以下是处理数据的整个过程：
        1. Read text file and split into lines, split lines into pairs
        2. Normalize text, filter by length and content
        3. Make word lists from sentences in pairs

'''

def prepareData(lang1, lang2, reverse=False):

    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))

    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))

    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])

    print("Counted words:")
    print(input_lang.name, input_lang.n_words)     # 打印源语言的名称与总词数(去重后的)
    print(output_lang.name, output_lang.n_words)   # 打印目标语言的名称与总词数(去重后的)

    return input_lang, output_lang, pairs

input_lang, output_lang, pairs = prepareData('setup', 'punchline')
print(random.choice(pairs))
print('\n')


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def variableFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    return Variable(torch.LongTensor(indexes).unsqueeze(0))


def variableFromPair(input_lang, output_lang, pair):
    input_variable = variableFromSentence(input_lang, pair[0])
    target_variable = variableFromSentence(output_lang, pair[1])
    return input_variable, target_variable



def index_and_pad(lang, dat):
    return sequence.pad_sequences([indexesFromSentence(lang, s)
                                  for s in dat], padding='post').astype(np.int64)

fra, eng = list(zip(*pairs))
fra = index_and_pad(input_lang, fra)
eng = index_and_pad(output_lang, eng)

def get_batch(x, y, batch_size=16):
    idxs = np.random.permutation(len(x))[:batch_size]
    return x[idxs], y[idxs]