from gensim.models import word2vec
import numpy as np
import pandas as pd
import jieba
import string

word_path = "/home/molly/code/moli_experiment/Hongloumeng/data_source/红楼梦.txt"
stop_path = "/home/molly/code/moli_experiment/Hongloumeng/data_source/stopwordsChinese.txt"
with open (word_path, "r") as myfile:
    data = myfile.read().splitlines()
with open (stop_path, "r") as myfile:
    stopwords = myfile.read().splitlines()


alpha = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
punc = "！？｡。＂＃＄％＆＇（）＊＋，\一－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
strange_str = ['    华语第一言情小说网站：红袖添香网（）为你提供最优质的言情小说在线阅读', '正文', '无弹窗广告', '小说网', ' ']


def search_similar(word:str=None, number:int=20):
    text = []
    for ch in data:
        if ch not in strange_str:
            for c in ch:
                if u'\u4e00' <= c and c <= u'\u9fff':
                    text.append(ch)
                    break

    df = pd.DataFrame({"paragraph":text})

    sentences = list(df["paragraph"].apply(sentence_preprocess))
    seg_list = []
    for sentence in sentences:
        sentence_cut = jieba.cut(sentence)
        final = []
        for seg in sentence_cut:
            if seg not in stopwords:
                final.append(seg)
        seg_list.append(" ".join(list(final)))

    sentences = [s.split() for s in seg_list]

    model = word2vec.Word2Vec(sentences, min_count=1)
    return model.wv.most_similar(word, topn=number)


def sentence_preprocess(sentence:str):
    # remove space, alphabet, digit and puctuation
    sentence = sentence.strip()
    sentence = "".join(char for char in sentence if not char.isdigit() and not char in alpha and not char in punc)
    for strange in strange_str:
        sentence = sentence.replace(strange, "")
    for punctuation in string.punctuation:
        sentence = sentence.replace(punctuation, "")
    return sentence
