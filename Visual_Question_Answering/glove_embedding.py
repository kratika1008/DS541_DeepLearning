import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import json
import deepdish as d
import os

from numpy import asarray
from numpy import zeros

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding



def read_files(path):
    #read the preprocess file
    preprocessed1_train=pd.read_csv(path+"/preprocessed1_whole.csv")
    glove_pretrained_weights = open(path+'glove.6B.300d.txt',encoding="utf8")
    #get question and answers
    questions=[]
    answers=[]
    questions=preprocessed1_train['question'].to_list()
    answers=preprocessed1_train['ans'].to_list()
    ques_id=preprocessed1_train['ques_id'].to_list()
    return(questions,answers,ques_id,glove_pretrained_weights)

def preprocess_question(questions):
    #get question vocab len
    t = Tokenizer()
    t.fit_on_texts(questions)
    vocab_size = len(t.word_index) + 1
    print(vocab_size)
    encoded_docs = t.texts_to_sequences(questions)
    return(t,vocab_size,encoded_docs)


def get_ans_vocab(answers):
    #get answer vocab and top answers 
    counts = {}
    for i in answers:
        ans = i
        counts[i] = counts.get(ans, 0) + 1
        
    cw = sorted([(count,w) for w,count in counts.items()], reverse=True)
    print ('top answer and their counts:')
    print ('\n'.join(map(str,cw[:20])))
    len(cw)
    
    ans_vocab = []
    for i in range(len(cw)):
        ans_vocab.append(cw[i][1])

    atoi = {w:i+1 for i,w in enumerate(ans_vocab)}
    itoa = {i+1:w for i,w in enumerate(ans_vocab)}
    ans_vocab_size = len(ans_vocab) + 1
    print(ans_vocab_size)
    N = len(answers)
    encoded_ans = np.zeros(N, dtype='uint32')
    for i in range(len(answers)):
        encoded_ans[i] = atoi[answers[i]]
    return(cw,ans_vocab,atoi,itoa,ans_vocab_size,encoded_ans,N)



def doc_padding(encoded_docs):
    padded_docs = pad_sequences(encoded_docs, maxlen=26, padding='post')
    return(padded_docs)



def glove_embedding(vocab_size,t,glove_pretrained_weights):
    embeddings_index = dict()
    for line in glove_pretrained_weights:
        values = line.split()
        word = values[0]
        coefs = asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    glove_pretrained_weights.close()
    embedding_matrix = zeros((vocab_size, 300))

    for word, i in t.word_index.items():
      embedding_vector = embeddings_index.get(word)
      #print(embedding_vector)
      if embedding_vector is not None:
    	  embedding_matrix[i] = embedding_vector
          
    return(embedding_matrix)
    
    

def main():
    
    path=os.getcwd()

    questions,answers,ques_id,glove_pretrained_weights=read_files(path)
    t,vocab_size,encoded_docs=preprocess_question(questions)
    cw,ans_vocab,atoi,itoa,ans_vocab_size,encoded_ans,N=get_ans_vocab(answers)
    padded_docs=doc_padding(encoded_docs)

    Quest_dict={}
    Ans_dict={}
    for i in range(len(padded_docs)):
        Quest_dict[ques_id[i]]=padded_docs[i]
        Ans_dict[ques_id[i]]=encoded_ans[i]
    
    embedding_matrix=glove_embedding(vocab_size)
    

    d.io.save('D:/MS DS WPI/Sem II/Deep Learning/project/Data/train_val/Quest_dict_9.h5',Quest_dict)
    d.io.save('D:/MS DS WPI/Sem II/Deep Learning/project/Data/train_val/Ans_dict_9.h5',Ans_dict)
    
    np.save("D:/MS DS WPI/Sem II/Deep Learning/project/Data/train_val/glove_q_doc",padded_docs)
    
    np.save("D:/MS DS WPI/Sem II/Deep Learning/project/Data/train_val/glove_a_doc",encoded_ans)
    
    np.save("D:/MS DS WPI/Sem II/Deep Learning/project/Data/train_val/embedding_matrix",embedding_matrix)
    
    with open('D:/MS DS WPI/Sem II/Deep Learning/project/Data/train_val/Question_i_9.json', 'w') as outfile:
        json.dump(t.word_index, outfile)
    
    with open('D:/MS DS WPI/Sem II/Deep Learning/project/Data/train_val/ansvocab_i_9.json', 'w') as fp:
        json.dump(atoi, fp)



